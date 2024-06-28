import spade
import yaml
from Agents.ClientAgent import FSMAgentClient
from Central.Central_Agent import Server
import paths

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)

async def main():
    number_of_clients = config["learning_configuration"]["number_of_clients"]
    agents = {}

    server = Server(config["client"]["jid_server"], "test_server")
    agents["server"] = server
    await server.start(auto_register=True)

    for i in range(number_of_clients):
        #name_of_client = "client_" + str(uuid.uuid4())[:6]
        name_of_client = "client" + str(i)
        print("agent: " + name_of_client)
        agent = FSMAgentClient(name_of_client + "@localhost", name_of_client)
        agents[name_of_client] = agent
        await agent.start(auto_register=True)



if __name__ == "__main__":
        spade.run(main())