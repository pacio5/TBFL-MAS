from spade.agent import Agent
from spade.behaviour import *
import yaml

with open("../../config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)


class Agent_Client(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state_client(State):
    async def run(self):
        print("-    This is the setup state")
        self.set_next_state(config["client"]["receive"])


class receive_state_client(State):
    async def run(self):
        global config
        print("-    This is the receive state")
        await asyncio.sleep(10)
        self.set_next_state(config["client"]["receive"])


class FSMAgentClient(Agent):
    async def setup(self):
        global config
        fsm = Agent_Client()
        fsm.add_state(name=config["client"]["set_up"], state=setup_state_client(), initial=True)
        fsm.add_state(name=config["client"]["receive"], state=receive_state_client())

        fsm.add_transition(config["client"]["set_up"], config["client"]["receive"])
        fsm.add_transition(config["client"]["receive"], config["client"]["receive"])

        self.add_behaviour(self.ClientBehaviour())
        self.add_behaviour(fsm)

    class ClientBehaviour(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's approve it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)

        async def run(self):
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available

            self.presence.set_available()
            # self.presence.subscribe(self.agent.jid_server)
            self.presence.subscribe("server@localhost")


async def agent(name_of_client):
    jid_server = "server@localhost"
    client = FSMAgentClient(name_of_client + "@localhost", name_of_client)
    client.jid_server = jid_server
    await client.start()
    time.sleep(1)

    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await client.stop()
