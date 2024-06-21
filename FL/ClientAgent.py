from spade.agent import Agent
from spade.behaviour import *
import spade
import config

class Agent_Client(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state_client(State):
    async def run(self):
        print("-    This is the setup state")
        ''' self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available

        self.presence.set_available()
        self.presence.subscribe(self.agent.jid_server)
        self.presence.subscribe("server@localhost")
        '''
        self.set_next_state(config.RECEIVE_STATE_CLIENT)

class receive_state_client(State):
    async def run(self):
        print("-    This is the receive state")
        time.sleep(2)
        self.set_next_state(config.RECEIVE_STATE_CLIENT)

class FSMAgentClient(Agent):
    async def setup(self):
        fsm = Agent_Client()
        fsm.add_state(name=config.SETUP_STATE_CLIENT, state=setup_state_client(), initial=True)
        fsm.add_state(name=config.RECEIVE_STATE_CLIENT, state=receive_state_client())

        fsm.add_transition(config.SETUP_STATE_CLIENT, config.RECEIVE_STATE_CLIENT)
        fsm.add_transition(config.RECEIVE_STATE_CLIENT, config.RECEIVE_STATE_CLIENT)

        self.add_behaviour(self.ClientBehav())
        self.add_behaviour(fsm)

    class ClientBehav(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's aprove it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)

        async def run(self):
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available

            self.presence.set_available()
            # self.presence.subscribe(self.agent.jid_server)
            self.presence.subscribe("server@localhost")


async def main():
    jid_server = "server@localhost"
    for i in range(10):
        client = FSMAgentClient("client"+str(i)+"@localhost", "client"+str(i))
        client.jid_server = jid_server
        await client.start()
        time.sleep(1)


    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await client.stop()

async def agent0():
    jid_server = "server@localhost"
    client = FSMAgentClient("client0@localhost", "client0")
    client.jid_server = jid_server
    await client.start()
    time.sleep(1)


    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await client.stop()

async def agent1():
    jid_server = "server@localhost"
    client = FSMAgentClient("client1@localhost", "client1")
    client.jid_server = jid_server
    await client.start()
    time.sleep(1)


    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await client.stop()
if __name__ == "__main__":
    spade.run(agent0())
    #spade.run(agent1())