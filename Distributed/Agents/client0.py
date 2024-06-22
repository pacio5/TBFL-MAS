import os
import spade
from Distributed.ClientAgent import agent

if __name__ == "__main__":
    name_of_client = os.path.basename(__file__)[:-3]
    spade.run(agent(name_of_client))
