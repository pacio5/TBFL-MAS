import asyncio
import loggingtestcase
from loggingtestcase import DisplayLogs
from main import main
from Utilities.Paths import config
import spade
import time
import traceback
import unittest
from Utilities.Paths import Paths
import yaml


class TestFSM(unittest.TestCase):
    @loggingtestcase.capturelogs('FSM', level='INFO', display_logs=DisplayLogs.ALWAYS)
    def test_creation_and_fsm_behaviour(self, logs):
        # define configuration
        fedavg = {"algorithm": "FedAvg", "global_epoch": 1, "number_of_client_agents": 1}

        # run agents for testing FSM with defined configuration
        try:
            spade.run(main(fedavg))
        except Exception:
            print(traceback.format_exc())

        # set variables
        averaged = 0
        client_agents_are_present = 0
        is_created = 0
        is_set_up = 0
        metrics_calculated = 0
        metrics_stored = 0
        predicted = 0
        received_from_client_agents = 0
        received_from_server_agent = 0
        sent_to_client_agents = 0
        sent_to_the_server = 0
        started = 0
        stopped = 0
        trained_global_model = 0
        trained_local_model = 0

        # look at the logs
        for log in logs.output:
            if config["options"]["algorithm"] != "ML":
                if "averaged" in log:
                    averaged += 1
                if "client agents are present" in log:
                    client_agents_are_present += 1
                if "received from client agents" in log:
                    received_from_client_agents += 1
                if "received from server agent" in log:
                    received_from_server_agent += 1
                if "sent to client agents" in log:
                    sent_to_client_agents += 1
                if "sent to the server" in log:
                    sent_to_the_server += 1
                if "trained local model" in log:
                    trained_local_model += 1
            else:
                if "trained global model" in log:
                    trained_global_model += 1
            if "is created" in log:
                is_created += 1
            if "is set up" in log:
                is_set_up += 1
            if "metrics calculated" in log:
                metrics_calculated += 1
            if "metrics stored" in log:
                metrics_stored += 1
            if "predicted" in log:
                predicted += 1
            if "started" in log:
                started += 1
            if "stopped" in log:
                stopped += 1

        # assert if the agents are created and run correctly
        if config["options"]["algorithm"] != "ML":
            self.assertGreaterEqual(averaged, 1)
            self.assertGreaterEqual(client_agents_are_present, 1)
            self.assertGreaterEqual(received_from_client_agents, 1)
            self.assertGreaterEqual(received_from_server_agent, 1)
            self.assertGreaterEqual(sent_to_client_agents, 1)
            self.assertGreaterEqual(sent_to_the_server, 1)
            self.assertGreaterEqual(trained_local_model, 1)
        else:
            self.assertGreaterEqual(trained_global_model, 1)
        self.assertGreaterEqual(is_created, 1)
        self.assertGreaterEqual(is_set_up, 1)
        self.assertGreaterEqual(metrics_calculated, 1)
        self.assertGreaterEqual(metrics_stored, 1)
        self.assertGreaterEqual(predicted, 1)
        self.assertGreaterEqual(started, 1)
        self.assertGreaterEqual(stopped, 1)


if __name__ == "__main__":
    unittest.main()
