import asyncio
from Behaviours.SubscriptionBehaviour import *
import logging
import loggingtestcase
from loggingtestcase import DisplayLogs
import spade
from spade.agent import Agent
import unittest
import uuid

logger = logging.getLogger("SubscriptionBehaviour")


# client agent for testing the subscription function
class ClientAgent(Agent):
    def __init__(self, client_name, client_password, server_name):
        super().__init__(client_name, client_password)
        self.server_name = server_name

    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(SubscriptionBehaviour(self.server_name + "@localhost"))

# server agent for testing the subscription function
class ServerAgent(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(SubscriptionBehaviour())

# run main function for testing subscription
async def main():
    server_name = "server_" + str(uuid.uuid4())[:6]
    server_password = str(uuid.uuid4())[:12]
    server = ServerAgent(server_name + "@localhost", server_password)
    await server.start(auto_register=True)
    logger.info(server_name + " is created")

    client_name = "client_" + str(uuid.uuid4())[:6]
    client_password = str(uuid.uuid4())[:12]
    client = ClientAgent(client_name + "@localhost", client_password, server_name)
    await client.start(auto_register=True)
    logger.info(client_name + " is created")

    await asyncio.sleep(10)

    await server.stop()
    await client.stop()


class TestSubscription(unittest.TestCase):
    @loggingtestcase.capturelogs('SubscriptionBehaviour', level='INFO', display_logs=DisplayLogs.ALWAYS)
    def test_creation_and_subscription_behaviour(self, logs):
        # run agents for testing subscription
        spade.run(main())

        # set variables
        accepted_subscription = 0
        approved_subscription = 0
        is_available = 0
        is_created = 0

        # control the logs after subscription
        for log in logs.output:
            if "accepted subscription" in log:
                accepted_subscription += 1
            if "approved subscription" in log:
                approved_subscription += 1
            if "is available" in log:
                is_available += 1
            if "is created" in log:
                is_created += 1

        # assert that the subscription works
        self.assertEqual(2, accepted_subscription)
        self.assertEqual(2, approved_subscription)
        self.assertEqual(4, is_available)
        self.assertEqual(2, is_created)


if __name__ == "__main__":
    unittest.main()
