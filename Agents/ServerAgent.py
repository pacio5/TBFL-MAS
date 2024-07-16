from Behaviours.StartAndEndFSMBehaviour import *
from Behaviours.SubscriptionBehaviour import *
from spade.agent import Agent
from States.ServerAgentStates import *
from Utilities.Paths import config


# Server agent
class ServerAgent(Agent):
    def __init__(self, name, password, args, batch_sizes_per_classes, random_seed_testing, random_seed_training):
        super().__init__(name, password)
        self.args = args
        self.batch_sizes_training_by_non_iid = batch_sizes_per_classes
        self.random_seed_testing = random_seed_testing
        self.random_seed_training = random_seed_training

    async def setup(self):
        print("Agent {} running".format(self.name))
        fsm = StartAndEndFSMBehaviour()

        # add states
        fsm.add_state(name=config["server"]["set_up"], state=SetUpState(), initial=True)
        fsm.add_state(name=config["server"]["train"], state=TrainState())
        fsm.add_state(name=config["server"]["control_agents_present"], state=ControlAgentsPresentState())
        fsm.add_state(name=config["server"]["send"], state=SendState())
        fsm.add_state(name=config["server"]["receive"], state=ReceiveState())
        fsm.add_state(name=config["server"]["avg"], state=AvgState())
        fsm.add_state(name=config["server"]["predict"], state=PredictState())
        fsm.add_state(name=config["server"]["calculate_metrics"], state=CalculateMetricsState())
        fsm.add_state(name=config["server"]["store_metrics"], state=StoreMetricsState())

        # add transitions
        fsm.add_transition(config["server"]["set_up"], config["server"]["train"])
        fsm.add_transition(config["server"]["train"], config["server"]["predict"])
        fsm.add_transition(config["server"]["set_up"], config["server"]["control_agents_present"])
        fsm.add_transition(config["server"]["control_agents_present"], config["server"]["control_agents_present"])
        fsm.add_transition(config["server"]["control_agents_present"], config["server"]["send"])
        fsm.add_transition(config["server"]["send"], config["server"]["receive"])
        fsm.add_transition(config["server"]["receive"], config["server"]["avg"])
        fsm.add_transition(config["server"]["avg"], config["server"]["predict"])
        fsm.add_transition(config["server"]["predict"], config["server"]["calculate_metrics"])
        fsm.add_transition(config["server"]["calculate_metrics"], config["server"]["control_agents_present"])
        fsm.add_transition(config["server"]["calculate_metrics"], config["server"]["train"])
        fsm.add_transition(config["server"]["calculate_metrics"], config["server"]["store_metrics"])

        # add behaviours
        self.add_behaviour(SubscriptionBehaviour())
        self.add_behaviour(fsm)
