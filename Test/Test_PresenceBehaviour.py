import asyncio
from Behaviours.Presence import *
import logging
import loggingtestcase
from loggingtestcase import DisplayLogs
import spade
from spade.agent import Agent
import unittest
import uuid

logger = logging.getLogger("PresenceBehaviour")


class Client(Agent):
    def __init__(self, client_name, client_password, server_name):
        super().__init__(client_name, client_password)
        self.server_name = server_name

    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(PresenceBehaviour(self.server_name + "@localhost"))

class Server(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(PresenceBehaviour())

async def main():
    server_name = "server_" + str(uuid.uuid4())[:6]
    server_password = str(uuid.uuid4())[:12]
    server = Server(server_name + "@localhost", server_password)
    await server.start(auto_register=True)
    logger.info(server_name + " is created")

    client_name = "client_" + str(uuid.uuid4())[:6]
    client_password = str(uuid.uuid4())[:12]
    client = Client(client_name + "@localhost", client_password, server_name)
    await client.start(auto_register=True)
    logger.info(client_name + " is created")

    await asyncio.sleep(10)

    await server.stop()
    await client.stop()


class TestPresenceBehaviour(unittest.TestCase):
    @loggingtestcase.capturelogs('PresenceBehaviour', level='INFO', display_logs=DisplayLogs.ALWAYS)
    def test_creation_and_presence_behaviour(self, logs):
        spade.run(main())
        accepted_subscription = 0
        approved_subscription = 0
        is_available = 0
        is_created = 0

        for log in logs.output:
            if "accepted subscription" in log:
                accepted_subscription += 1
            if "approved subscription" in log:
                approved_subscription += 1
            if "is available" in log:
                is_available += 1
            if "is created" in log:
                is_created += 1

        self.assertEqual(2, accepted_subscription)
        self.assertEqual(2, approved_subscription)
        self.assertEqual(4, is_available)
        self.assertEqual(2, is_created)


if __name__ == "__main__":
    unittest.main()
