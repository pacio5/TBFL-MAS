import asyncio
import loggingtestcase
from loggingtestcase import DisplayLogs
from main import main
from Utilities.Paths import config
import spade
import time
import traceback
import unittest
import yaml


class TestFSM(unittest.TestCase):
    @loggingtestcase.capturelogs('FSM', level='INFO', display_logs=DisplayLogs.ALWAYS)
    def test_creation_and_fsm_behaviour(self, logs):
        try:
            spade.run(main())
        except Exception:
            print(traceback.format_exc())

        # set variables
        averaged = 0
        clients_are_present = 0
        finished = 0
        is_created = 0
        is_set_up = 0
        metrics_calculated = 0
        metrics_stored = 0
        predicted = 0
        received_from_clients = 0
        received_from_server_agent = 0
        sent_to_clients = 0
        sent_to_the_server = 0
        starting = 0
        trained_global_model = 0
        trained_local_model = 0

        # look at the logs
        for log in logs.output:
            if config["options"]["algorithm"] != "ML":
                if "averaged" in log:
                    averaged += 1
                if "clients are present" in log:
                    clients_are_present += 1
                if "received from clients" in log:
                    received_from_clients += 1
                if "received from server agent" in log:
                    received_from_server_agent += 1
                if "sent to clients" in log:
                    sent_to_clients += 1
                if "sent to the server" in log:
                    sent_to_the_server += 1
                if "trained local model" in log:
                    trained_local_model += 1
            else:
                if "trained global model" in log:
                    trained_global_model += 1
            if "finished" in log:
                finished += 1
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
            if "starting" in log:
                starting += 1

        # assert if the agents are created and run correctly
        if config["options"]["algorithm"] != "ML":
            self.assertGreaterEqual(1, averaged)
            self.assertGreaterEqual(1, clients_are_present)
            self.assertGreaterEqual(1, received_from_clients)
            self.assertGreaterEqual(1, received_from_server_agent)
            self.assertGreaterEqual(1, sent_to_clients)
            self.assertGreaterEqual(1, sent_to_the_server)
            self.assertGreaterEqual(1, trained_local_model)
        else:
            self.assertGreaterEqual(1, trained_global_model)
        self.assertGreaterEqual(1, finished)
        self.assertGreaterEqual(1, is_created)
        self.assertGreaterEqual(1, is_set_up)
        self.assertGreaterEqual(1, metrics_calculated)
        self.assertGreaterEqual(1, metrics_stored)
        self.assertGreaterEqual(1, predicted)
        self.assertGreaterEqual(1, starting)


if __name__ == "__main__":
    unittest.main()
