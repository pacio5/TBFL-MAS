from Agents.Central_Agent import Server
from Agents.Client_Agent import Client
import logging
import paths
import spade
import yaml

fsm_logger = logging.getLogger("FSM")

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)

async def main():
    number_of_clients = config["learning_configuration"]["FL"]["number_of_clients"]
    agents = {}

    server = Server(config["client"]["jid_server"], "test_server")
    agents["server"] = server
    await server.start(auto_register=True)
    fsm_logger.info(config["client"]["jid_server"] + " is created")
    if config["learning_configuration"]["FL_or_ML"] == "FL":
        for i in range(number_of_clients):
            #name_of_client = "client_" + str(uuid.uuid4())[:6]
            name_of_client = "client" + str(i)
            print("agent: " + name_of_client)
            agent = Client(name_of_client + "@localhost", name_of_client)
            agents[name_of_client] = agent
            await agent.start(auto_register=True)
            fsm_logger.info(name_of_client + " is created")


if __name__ == "__main__":
        spade.run(main())