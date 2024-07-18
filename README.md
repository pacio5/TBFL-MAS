# Federated Learning with Multi-Agent System

This project is about implementing Federated Learning with Multi-Agent System with the frameworks SPADE and PyTorch.

## Configuration
The configuration files are all in the Configuration folder.
The default configurations are configured in conf.yml file.
The optional learning scenarios are configured in learning_scenarios_config.yml file.
The selection of the learning scenarios that should run, are configured in launch_config.yml file.

## Arguments
The configuration of the program can also be influenced through arguments. 
This is configured in the Argparser.py in folder Utilities.

### example run configuration
The argument launch_config takes a list of string for defining the learning scenarios.
Here in an example for the learning scenarios FedAvg and FedSGD:
{path to python exe}\python.exe {path to main.py}\main.py --launch_config FedAvg FedSGD

## Run program
The program can be run from main.py.

## Plot
The pictures can be plotted by the plot.py file.
Depending on which posts you want to see, you have to adjust the plot function.
