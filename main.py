from Agents.ServerAgent import ServerAgent
from Agents.ClientAgent import ClientAgent
from Utilities.Argparser import Argparser
import asyncio
from Utilities.Data import Data
import logging
from Utilities.Metrics import Metrics
import os
from Utilities.Paths import config, Paths
import spade
import uuid
import yaml

fsm_logger = logging.getLogger("FSM")
agents = {}

async def main(config_file=""):
    args = Argparser.args_parser()
    batch_size_options = config["options"]["batch_size_per_class"]
    if config_file != "":
        args.config_file = config_file

    opt = vars(args)
    yaml_conf = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
    opt.update(yaml_conf)

    batch_size_per_classes_server = {}
    Data.prepare_batch_sizes_per_classes(args, batch_size_options,
                                    batch_size_per_classes_server, 100)

    server = ServerAgent(args.jid_server, "server", args,
                         batch_size_per_classes_server, 100, 200)
    agents["server"] = server
    await server.start(auto_register=True)
    fsm_logger.info(args.jid_server + " is created")
    if args.algorithm != "ML":
        for i in range(args.number_of_agents):
            batch_size_per_classes = {}
            Data.prepare_batch_sizes_per_classes(args, batch_size_options, batch_size_per_classes, i * 13)
            # name_of_client = "client_" + str(uuid.uuid4())[:6]
            name_of_client = "client" + str(i)
            agent = ClientAgent(name_of_client + "@localhost", name_of_client, args,
                                batch_size_per_classes, i * 7, i * 9)
            agents[name_of_client] = agent
            await agent.start(auto_register=True)
            fsm_logger.info(name_of_client + " is created")

    timer = 0
    while agents["server"].is_alive():
        try:
            await asyncio.sleep(1)
            # simulate new entry of an agent
            if timer == 20 and args.new_entry_or_leave == "new entry":
                batch_size_per_classes = {}
                Data.prepare_batch_sizes_per_classes(args, batch_size_options, batch_size_per_classes, 20)
                name_of_client = "client_" + str(uuid.uuid4())[:6]
                print("agent: " + name_of_client)
                agent = ClientAgent(name_of_client + "@localhost", name_of_client, args,
                                    batch_size_per_classes, 20, 30)
                agents[name_of_client] = agent
                await agent.start(auto_register=True)
                fsm_logger.info(name_of_client + " is created")
            # simulate leave of an agent
            elif timer == 35 and args.new_entry_or_leave == "leave":
                agents["client0"].presence.unsubscribe(args.jid_server)
                print("client0 leaves the MAS")
                fsm_logger.info("client0 leaves the MAS")
            timer += 1
        except KeyboardInterrupt:
            break

async def multiple_mains():
    path_to_files = str(Paths.get_project_root()) + "\\Configuration\\RunConfiguration\\"
    for file in os.listdir(path_to_files):
        await main(config_file=path_to_files + file)
        print(file + " is done")



def plot():
    for key in agents.keys():
        print("plotting: " + key)
        Metrics.plot_metrics(key, key, "test")


if __name__ == "__main__":
    spade.run(multiple_mains())
    plot()