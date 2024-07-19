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
With the plot.py file you can show some plots.
If you want to plot all comparisons of the fourteen learning scenarios, just run the following command. The default value of argument plot_mode is already zero, no adjustment is needed.
```
{path to python exe}\python.exe {path to plot.py}\plot.py
```
If you want to customize a plot, you have to set the argument plot_mode to one.
```
{path to python exe}\python.exe {path to plot.py}\plot.py --plot_mode 1
```
An example of plotting a customized plot:
```
{path to python exe}\python.exe {path to plot.py}\plot.py --plot_mode 1 --learning_scenarios_to_plot FedSGD --title_learning_scenario_to_plot FedSGD: --metrics_to_plot test_f1 --title_metrics_to_plot f1 --ylabel_to_plot f1-score 
```
