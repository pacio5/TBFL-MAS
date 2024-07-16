from Behaviours.StartAndEndFSMBehaviour import *
from Behaviours.SubscriptionBehaviour import *
from States.ClientAgentStates import *
from spade.agent import Agent


# Client agent
class ClientAgent(Agent):
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
        fsm.add_state(name=config["client_agent"]["set_up"], state=SetUpState(), initial=True)
        fsm.add_state(name=config["client_agent"]["receive"], state=ReceiveState())
        fsm.add_state(name=config["client_agent"]["train"], state=TrainState())
        fsm.add_state(name=config["client_agent"]["send"], state=SendState())
        fsm.add_state(name=config["client_agent"]["predict"], state=PredictState())
        fsm.add_state(name=config["client_agent"]["calculate_metrics"], state=CalculateMetricsState())
        fsm.add_state(name=config["client_agent"]["store_metrics"], state=StoreMetricsState())

        # add transitions
        fsm.add_transition(config["client_agent"]["set_up"], config["client_agent"]["receive"])
        fsm.add_transition(config["client_agent"]["receive"], config["client_agent"]["receive"])
        fsm.add_transition(config["client_agent"]["receive"], config["client_agent"]["train"])
        fsm.add_transition(config["client_agent"]["train"], config["client_agent"]["send"])
        fsm.add_transition(config["client_agent"]["send"], config["client_agent"]["predict"])
        fsm.add_transition(config["client_agent"]["predict"], config["client_agent"]["calculate_metrics"])
        fsm.add_transition(config["client_agent"]["calculate_metrics"], config["client_agent"]["receive"])
        fsm.add_transition(config["client_agent"]["calculate_metrics"], config["client_agent"]["store_metrics"])

        # add behaviours
        self.add_behaviour(SubscriptionBehaviour(self.args.jid_server))
        self.add_behaviour(fsm)
