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

## Prosody configuration with Docker 

Create a prosody container:
```
docker run -d --name prosody -p 5222:5222 prosody/prosody
```


You can generate a self-signed certificate using openssl:
```
openssl req -new -x509 -days 365 -nodes -out /etc/prosody/certs/localhost.crt -keyout /etc/prosody/certs/localhost.key
```
Make sure to replace localhost with the appropriate domain name if you are not using localhost.

Open your Prosody configuration file, typically located at /etc/prosody/prosody.cfg.lua.
Add or update the following lines in your configuration file to point to your certificate and key files:
```
-- Replace with your domain and certificate paths
VirtualHost "localhost"
    ssl = {
        key = "/etc/prosody/certs/localhost.key";
        certificate = "/etc/prosody/certs/localhost.crt";
    }
```

Ensure that Prosody requires TLS for client-to-server (c2s) connections:
```
c2s_require_encryption = true
s2s_require_encryption = true
```

Ensure that Prosody can read the certificate and key files:
```
chown prosody:prosody /etc/prosody/certs/localhost.key /etc/prosody/certs/localhost.crt
chmod 600 /etc/prosody/certs/localhost.key /etc/prosody/certs/localhost.crt
```
After making these changes, restart the Prosody service:
```
prosodyctl restart
```

Note: during restart the container may shut down, in case it should be started manually


# Tests

**unittest** can be used to run the tests. 

To perform all available tests, run the following command:
```
python -m unittest discover -s Test 
```
where Test is the folder containing the tests.

To perform a specific test, run the following command:
```
python -m unittest Test.test_file_name
```

If you want to create new tests, you need to use the notation test_testname.py 

