# Federated Learning with Multi-Agent System

This project is about implementing Federated Learning with Multi-Agent System with the frameworks SPADE and PyTorch.

## Project Structure

The project is structured as follows:

- **Agents**: Contains the agents that are used in the MAS, such as ServerAgent and ClientAgent.
- **Configuration**: Contains the configuration files for the learning scenarios.
- **fashion-dataset**: Contains the dataset that is used as benchmark.
- **States**: Contains the states that are used in the FL-MAS, for both client and server agents.
- **Tests**: Contains the tests to check the functionality of the code.
- **Utilities**: Contains class and functions that are used in the project to read parameters, plot, store metrics and more.

## Instructions for running the code

There are two ways to run the code. The first way is to run the code with the local environment. 
The second way is to run the code with Docker.

### Local Environment

To run the code with the local environment, you need to install the requirements: 
- python >= 3.11 
- pip >= 24.2
- cuda drivers to run the code on GPU

After installing the requirements, you need to install the necessary python libraries:

```
pip install -r requirements.txt
```

Then you need to install and configure an XMPP server.
Then you need to install and configure an XMPP server. In this case, Prosody's configuration with Docker is presented. Naturally, any XMPP server is valid.

#### Prosody configuration with Docker 

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

Allow clients registration:
```
allow_registration = true
modules_enabled = { \"register\" }
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

#### Running the code
To run the code, you need to run the main.py file with some (optional) arguments: 

- agents_to_plot: filter agents to plot.
- algorithm: algorithm that is used for learning, default is FedAvg.
- batch_size_testing: batch size for testing.
- batch_size_training: batch size for training.
- dataset_testing: path to the dataset for testing.
- dataset_training: path to the dataset for training.
- epoch: current epoch.
- global_epochs: number of global epochs.
- iid: default is IID, if you want to run non-IID set to 0.
- jid_server: address of jid server.
- learning_scenarios_to_plot: filter learning scenarios to plot.
- launch_config: The learning scenarios that should run. This argument takes a list of string for defining the learning scenarios.
- learning_rate: learning rate for the optimizer.
- local_epochs: number of local epochs.
- metrics_to_plot: filter metrics to plot.
- new_entry_or_leave: default is no new entry or leave.
- number_of_client_agents: number of client agents.
- number_of_classes_in_dataset: number of classes in the dataset.
- plot_mode: 0 for all comparisons, 1 for customized plot.
- standard_deviation_for_noise: standard deviation for noise.
- title_learning_scenario_to_plot: title for learning scenario to plot.
- title_metrics_to_plot: title for metrics to plot.
- wait_until_threshold_is_reached: default is no threshold.
- threshold: threshold needed to reach before learning ends. 
- xlabel_to_plot: x label for plot.
- ylabel_to_plot: y label for plot.

Default values are set in the config.yml file.
The optional learning scenarios are configured in learning_scenarios_config.yml file.
The selection of the learning scenarios that should run, are configured in launch_config.yml file.


Here in an example to run the code:
```
python main.py --launch_config FedAvg FedSGD
```

After running the code, the results are stored in the Results folder.


### Docker

To run the code with Docker, you need to install Docker and Docker Compose. There are three files in the project that handle execution with Docker: 
- Dockerfile.gpu: Dockerfile for GPU execution.
- Dockerfile.prosody: Dockerfile for Prosody configuration.
- docker-compose.yml: Docker Compose file to run the code.

Cuda drivers must be installed on the machine to run the project.

Results are stored in the Results in the host machine, you can set up the path in the docker-compose.yml file.

Example to run the code with Docker:

```
docker compose up -d --build fl-mas > {path_to_store_on_host_machine}/fl-mas-output.txt
```

After running the code, containers aren't stopped. To stop and remove containers and images, run the following command in the folder where the docker-compose.yml file is located:

```
docker compose down
docker container rm {container_id}
docker rmi {image_id}
```

## Test

To run the tests, it is necessary to have the local environment configured. 

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

If you want to create new tests, you need to use the notation **test_testname.py**

## Plot

To plot the results, only python is needed.

With the plot.py file you can show some plots.
If you want to plot all comparisons of the fourteen learning scenarios, just run the following command. The default value of argument plot_mode is already zero, no adjustment is needed.
```
python plot.py
```
If you want to customize a plot, you have to set the argument plot_mode to one.
```
python plot.py --plot_mode 1
```
An example of plotting a customized plot:
```
python plot.py --plot_mode 1 --learning_scenarios_to_plot FedSGD --title_learning_scenario_to_plot FedSGD: --metrics_to_plot test_f1 --title_metrics_to_plot f1 --ylabel_to_plot f1-score 
```
