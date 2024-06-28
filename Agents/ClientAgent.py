from States.Client_States import *
from Behaviours import *

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)


class FSMAgentClient(Agent):
    async def setup(self):
        fsm = ExtendedFSMBehaviour()
        fsm.add_state(name=config["client"]["set_up"], state=set_up_state(), initial=True)
        fsm.add_state(name=config["client"]["receive"], state=receive_state())
        fsm.add_state(name=config["client"]["train"], state=train_state())
        fsm.add_state(name=config["client"]["send"], state=send_state())

        fsm.add_transition(config["client"]["set_up"], config["client"]["receive"])
        fsm.add_transition(config["client"]["receive"], config["client"]["receive"])
        fsm.add_transition(config["client"]["receive"], config["client"]["train"])
        fsm.add_transition(config["client"]["train"], config["client"]["send"])
        fsm.add_transition(config["client"]["send"], config["client"]["receive"])

        self.add_behaviour(PresenceBehaviour(config["client"]["jid_server"]))
        self.add_behaviour(fsm)
