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

## Example run
The argument launch_config takes a list of string for defining the learning scenarios.
Here in an example for the learning scenarios FedAvg and FedSGD:
```
{path to python exe}\python.exe {path to main.py}\main.py --launch_config FedAvg FedSGD
```

## Plot
All learning scenarios from 1 to 14 can be plotted by the plot.py file.
If one wants to see something else, one needs to adjust the plot function.
```
{path to python exe}\python.exe {path to plot.py}\plot.py
```