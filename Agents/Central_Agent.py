from Behaviours.ExtendedFSM import *
from Behaviours.Presence import *
from spade.agent import Agent
from States.Server_States import *

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)


class Server(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        fsm = ExtendedFSMBehaviour()

        fsm.add_state(name=config["server"]["set_up"], state=set_up_state(), initial=True)
        fsm.add_state(name=config["server"]["train"], state=train_state())
        fsm.add_state(name=config["server"]["control_agents_present"], state=control_agents_present_state())
        fsm.add_state(name=config["server"]["send"], state=send_state())
        fsm.add_state(name=config["server"]["receive"], state=receive_state())
        fsm.add_state(name=config["server"]["avg"], state=avg_state())
        fsm.add_state(name=config["server"]["predict"], state=predict_state())
        fsm.add_state(name=config["server"]["collect_metrics"], state=collect_metrics_state())
        fsm.add_state(name=config["server"]["plot_metrics"], state=plot_metrics_state())
        fsm.add_state(name=config["server"]["finish"], state=finish_state())

        fsm.add_transition(config["server"]["set_up"], config["server"]["train"])
        fsm.add_transition(config["server"]["train"], config["server"]["predict"])
        fsm.add_transition(config["server"]["set_up"], config["server"]["control_agents_present"])
        fsm.add_transition(config["server"]["control_agents_present"], config["server"]["control_agents_present"])
        fsm.add_transition(config["server"]["control_agents_present"], config["server"]["send"])
        fsm.add_transition(config["server"]["send"], config["server"]["receive"])
        fsm.add_transition(config["server"]["receive"], config["server"]["avg"])
        fsm.add_transition(config["server"]["avg"], config["server"]["predict"])
        fsm.add_transition(config["server"]["predict"], config["server"]["collect_metrics"])
        fsm.add_transition(config["server"]["collect_metrics"], config["server"]["send"])
        fsm.add_transition(config["server"]["collect_metrics"], config["server"]["train"])
        fsm.add_transition(config["server"]["collect_metrics"], config["server"]["plot_metrics"])
        fsm.add_transition(config["server"]["plot_metrics"],config["server"]["finish"])

        self.add_behaviour(PresenceBehaviour())
        self.add_behaviour(fsm)
