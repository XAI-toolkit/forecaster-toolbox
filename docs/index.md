# Forecasting Toolbox

## Description

This repository contains the source code of the **Forecasting Toolbox back-end**, which is part of the [SDK4ED Platform](https://sdk4ed.eu/). The purpose of the Forecasting Toolbox is to provide predictive forecasts regarding the evolution of the three core quality attributes targeted by the SDK4ED platform, namely *Technical Debt*, *Energy* and *Dependability (Security)*. The entry point of the Forecasting Toolbox is a RESTful web server that uses the [Flask web framework](https://www.palletsprojects.com/p/flask/) wrapped inside [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/), a Python WSGI production-ready server. At a lower level, the server exposes three sub-modules, implemented as individual web services. Each web service plays the role of an end-point that allows the invocation of a set of forecasting models and returns the results, providing users with insightful information for the future evolution each of the three core quality attributes of a software application. The services supported by the Forecasting Toolbox are listed below:
- **TD Forecaster**: This web service is responsible for generating Technical Debt forecasts for a given software application. A TD forecast represents the predicted evolution of the total remediation effort (measured in minutes) to fix all code issues (e.g. code smells, bugs, code duplications, etc.) of a software application, up to a future point specified by the user.
- **Energy Forecaster**: This web service is responsible for generating Energy forecasts for a given software application. An Energy forecast represents the predicted evolution of the total energy consumption (measured in Joules) of a software application, up to a future point specified by the user.
- **Dependability Forecaster**: This web service is responsible for generating Security forecasts for a given software application. A Security forecast represents the predicted evolution of the Security Index (value between 0 and 1 that aggregates the entire program security characteristics) of a software application, up to a future point specified by the user.

The three web services allow the individual and remote invocation of the forecasting models developed for estimating the evolution of TD, Energy and Security. This is achieved through the dedicated API exposed by the RESTful web server, which allows the user to perform simple HTTP GET requests to the three web services. Several inputs need to be provided as URL-encoded parameters to these requests. These parameters are listed below:

|   Parameter  |                           Description                           | Required |                                                                                                                                                                Valid Inputs                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|:------------:|:---------------------------------------------------------------:|:--------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| horizon      | The forecasting horizon up to which forecasts will be produced. |    Yes   | An integer in range [1-N], where N depends on the volume of data used to train the regressor. Currently there is no upper limit and the service returns an error if this value is set too high.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| project      | The project ID for which the forecasts will be produced.        |    Yes   | A string value representing the ID of the selected project for which a forecast was requested. This ID is used to retrieve the TD, Energy and Dependability analysis metrics from the corresponding Toolboxes’ DBs, which will then be used for forecasting model execution. Depending on the specific web service, this ID is constructed as follows:<br>· TD Forecaster: ‘<user_name>:<project_name>’<br>· Security Forecaster: ‘<user_name>:<project_name>’<br>· Energy Forecaster: ‘<project_name>’<br>Both <user_name> and <project_name> values are project properties retrieved from the SDK4ED Dashboard session storage. |
| regressor    | The regressor model that will be used to produce forecasts.     |    No    | One of the following string values: [‘auto’, ‘mlr’, ‘lasso’, ‘ridge’, ‘svr_linear’, ‘svr_rbf’, ‘random_forest’, ‘arima’].<br>Default value is ‘auto’. If this parameter is omitted, default value is set to ‘auto’ and the service selects automatically the best model based on validation error minimization.                                                                                                                                                                                                                                                                                                                   |
| ground_truth | If the model will return also ground truth values or not.       |    No    | One of the following string values: [‘yes’, ‘no’].<br>Default value is ‘no.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| test         | If the model will produce Train-Test or unseen forecasts.       |    No    | One of the following string values: [‘yes’, ‘no’].<br>Default value is ‘no’. If set to ‘no’, then the service uses the whole data to train a regressor and returns forecasts on unseen data. A value of ‘yes’ should be used only for model testing and not actual deployment into production.                                                                                                                                                                                                                                                                                                                                    |

The output of the three individual web services provided by the Forecasting Toolbox, namely TD Forecaster, Energy Forecaster and Dependability Forecaster is a JSON file containing the predicted values for a particular quality attribute of the selected application. This JSON actually contains i) a status code of the response, ii) a N-size array containing the forecasts, where N is equal to the ‘horizon’ parameter, iii) a recap on the given parameter values, and iv) a message informing the user if the request was fulfilled successfully or not.

## Installation

### Installation using Anaconda

In this section, we provide instructions on how the user can build the python Flask server of the Forecasting Toolbox from scratch, using the Anaconda virtual environment. The Forecasting Toolbox is developed to run on Unix and Windows systems with python 3.6.* innstalled. We suggest installing python via the Anaconda distribution as it provides an easy way to create a virtual environment and install dependencies. The configuration steps needed, are described below:

- **Step 1**: Download the latest [Anaconda distribution](https://www.anaconda.com/distribution/) and follow the installation steps described in the [Anaconda documentation](https://docs.anaconda.com/anaconda/install/windows/).
- **Step 2**: Open Anaconda cmd. Running Anaconda cmd activates the base environment. We need to create a specific environment to run Forecasting Toolbox. Create a new python 3.6.4 environment by running the following command:
```bash
conda create --name forecaster_toolbox python=3.6.4
```
This command will result in the creation of a conda environment named *forecaster_toolbox*. In order to activate the new environment, execute the following command:
```bash
conda activate forecaster_toolbox
```
- **Step 3**: Once your newly created environment is active, install the needed libraries by executing the following commands:
```bash
conda install -c anaconda numpy pandas scikit-learn waitress flask flask-cors pymongo
```
and
```bash
conda install -c saravji pmdarima
```
- **Step 4**: To start the server, use the command promt inside the active environment and execute the commands described in section [Run Server](/sdk4ed/forecaster-toolbox#run-built-in-server).

### Installation using Docker

In this section, we provide instructions on how the user can build a new Docker Image that contains the python Flask app and the Conda environment of the of the Forecasting Toolbox. We highly recommend the users to select this way of installing the SDK4ED Forecasting Toolbox, as it constitutes the easiest way.

- **Step 1**: Download and install [Docker](https://www.docker.com/)
- **Step 2**: Clone the latest Forecasting Toolbox version and navigate to the home directory. You should see a [DockerFile](/blob/master/Dockerfile) and a [environment.yml](./blob/master/environment.yml) file, which contains the Conda environment dependencies. 
- **Step 3**: In the home directory of the Forecasting Toolbox, open cmd and execute the following command:
```bash
sudo docker build -t forecaster_toolbox .
``` 
This command will result in the creation of a Docker Image named *forecaster_toolbox*. In order to create a Docker Container from this image, execute the following command:
```bash
sudo docker run -it --name forecaster-toolbox-test -p 5000:5000 forecaster_toolbox
``` 
This command will generate and run a Docker Container named *forecaster-toolbox-test* in interactive session mode, i.e. it will open a command promt inside the Container. 
- **Step 4**: To start the server, use the command promt inside the running Container and execute the commands described in section [Run Server](/sdk4ed/forecaster-toolbox#run-built-in-server).

### Installation of the Database

Since the TD, Energy and Dependability forecasts are produced "on the fly", the Forecasting Toolbox does not require a running database instance to be functional. However, in case you require access to previously produced forecasting results, a database dedicated to store the output of the Forecasting web services might be of help. In that case, MongoDB is a well-suited option for the purposes of the Forecasting Toolbox.

To quickly install a MongoDB using Docker, open cmd and execute the following command:
```bash
sudo docker run --detach  \
  -p 27017:27017  \
  --name mongodb  \
  --volume /home/<user_name>/Desktop/mongo_data:/data/db  \
  mongo
```
This command will generate and run a MongoDB Docker Container named *mongodb*, which will serve as the Forecasting Toolbox dedicated DB.

## Run Server

You can run the server in various modes using Python to run the `forecaster_service.py` script:

```
usage: forecaster_service.py [-h] [-dh DB_HOST] [-dp DB_PORT] [-dn DB_DBNAME]
                             [--debug]
                             HOST PORT SERVER_MODE

positional arguments:
  HOST           Server HOST (e.g. "localhost")
  PORT           Server PORT (e.g. "5000")
  SERVER_MODE    builtin, waitress

optional arguments:
  -h, --help     show this help message and exit
  -dh DB_HOST    MongoDB HOST (e.g. "localhost") (default: localhost)
  -dp DB_PORT    MongoDB PORT (e.g. "27017") (default: 27017)
  -dn DB_DBNAME  Database NAME (default: forecaster_service)
  --debug        Run builtin server in debug mode (default: False)
```

`HOST`, `PORT`, and `SERVER_MODE` arguments are **mandatory**. You can set them according to your needs.

`DB_HOST`, `DB_PORT`, and `DB_DBNAME` arguments are **optional** and assume that there is a MongoDB instance running either on a local machine or remotely. In case that there is no such MongoDB instance running, the Forecasting Toolbox will still return the results, but they will not be stored anywhere.

### Run built-in Flask server

```
         127.0.0.1:5000
Client <----------------> Flask
```

To start the Forecasting Toolbox using the built-in **Flask** server, use the command promt inside the active Conda or Container environment and execute the following command: 

```bash
python forecaster_service.py 0.0.0.0 5000 builtin --debug
```

This command will start the built-in Flask server locally (0.0.0.0) on port 5000.

**MongoDB Integration**

In case there is a MongoDB instance running, use the command promt inside the active conda or Container environment and execute the following command: 

```bash
python forecaster_service.py 0.0.0.0 5000 builtin -dh localhost -dp 27017 -dn forecaster_service --debug
```

This command will start the built-in Flask server locally on port 5000 and store the results on a MongoDB database named "forecaster_service" running locally on port 27017.

**Warning**: The built-in Flask mode is useful for development since it has debugging enabled (e.g. in case of error the client gets a full stack trace). However, it is single-threaded. Do NOT use this mode in production!

### Run Waitress server

```
         127.0.0.1:5000
Client <----------------> Waitress <---> Flask
```

To start the Forecasting Toolbox using the **Waitress** server, use the command promt inside the active Conda or Container environment and execute the following command:

```bash
python forecaster_service.py 0.0.0.0 5000 waitress
```

This command will start the Waitress server locally (0.0.0.0) on port 5000.

**MongoDB Integration**

In case there is a MongoDB instance running, use the command promt inside the active conda or Container environment and execute the following command: 

```bash
python forecaster_service.py 0.0.0.0 5000 waitress -dh localhost -dp 27017 -dn forecaster_service
```

This command will start the Waitress server locally on port 5000 and store the results on a MongoDB database named "forecaster_service" running locally on port 27017.

**Warning**: The Waitress mode is higly recommended in real production environments, since it supports scaling and multiple-request handling features.

## Run Tests

A series of dedicated tests have been developed using the [pytest](https://docs.pytest.org/en/latest/) framework in order to ensure the proper execution of the Forecasting Toolbox. Once the server is installed, the user can run the testing suite by opening a new command promt inside the active Conda or Container environment and executing the following command:

```bash
pytest -v
```

A list of results will start popping on the command prompt, informing the user whether a test has PASSED or FAILED.

## Usage

### Example

Once the server is running, open your web browser and navigate to the following URL:

http://127.0.0.1:5000/ForecasterToolbox/TDForecasting?horizon=5&project=apache_kafka&regressor=ridge&ground_truth=no&test=no

You will get a JSON response containing TD forecasts of a sample application (Apache Kafka) for an horizon of 5 versions ahead, using the Ridge regressor model.