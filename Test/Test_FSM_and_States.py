import asyncio
import loggingtestcase
from loggingtestcase import DisplayLogs
from main import main
import paths
import spade
import time
import traceback
import unittest
import yaml

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)

async def wait(x):
    print(x)
    await asyncio.sleep(x)

class TestFSM_States(unittest.TestCase):
    @loggingtestcase.capturelogs('FSM', level='INFO', display_logs=DisplayLogs.ALWAYS)
    def test_creation_and_fsm_behaviour(self, logs):
        try:
            spade.run(main())
        except Exception:
            print(traceback.format_exc())
        averaged = 0
        clients_are_present = 0
        finished = 0
        is_created = 0
        is_set_up = 0
        metrics_collected = 0
        metrics_plotted = 0
        predicted = 0
        received_from_clients = 0
        received_from_server_agent = 0
        sent_to_clients = 0
        sent_to_the_server = 0
        trained_global_model = 0
        trained_local_model = 0
        for log in logs.output:
            if config["learning_configuration"]["FL_or_ML"] == "FL":
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
            if "metrics collected" in log:
                metrics_collected += 1
            if "metrics plotted" in log:
                metrics_plotted += 1
            if "predicted" in log:
                predicted += 1
        if config["learning_configuration"]["FL_or_ML"] == "FL":
            self.assertGreaterEqual(averaged, 1)
            self.assertGreaterEqual(clients_are_present, 1)
            self.assertGreaterEqual(received_from_clients, 1)
            self.assertGreaterEqual(received_from_server_agent, 1)
            self.assertGreaterEqual(sent_to_clients, 1)
            self.assertGreaterEqual(sent_to_the_server, 1)
            self.assertGreaterEqual(trained_local_model, 1)
        else:
            self.assertGreaterEqual(trained_global_model, 1)
        self.assertGreaterEqual(finished, 1)
        self.assertGreaterEqual(is_created, 1)
        self.assertGreaterEqual(is_set_up, 1)
        self.assertGreaterEqual(metrics_collected, 1)
        self.assertGreaterEqual(metrics_plotted, 1)
        self.assertGreaterEqual(predicted, 1)


if __name__ == "__main__":
    unittest.main()
