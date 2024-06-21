from spade.agent import Agent
from spade.behaviour import *
import spade
import config

class Agent_Server(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state_server(State):
    async def run(self):
        print("-    This is the setup state")
        '''self.agent.presence.set_available()
        self.agent.presence.on_subscribe = self.on_subscribe
        self.agent.presence.on_subscribed = self.on_subscribed
        self.agent.presence.on_available = self.on_available'''
        self.set_next_state(config.RECEIVE_STATE_SERVER)

class receive_state_server(State):
    async def run(self):
        print("-    This is the receive state")
        time.sleep(2)
        self.set_next_state(config.RECEIVE_STATE_SERVER)

class FSMAgentServer(Agent):
    async def setup(self):
        fsm = Agent_Server()
        fsm.add_state(name=config.SETUP_STATE_SERVER, state=setup_state_server(), initial=True)
        fsm.add_state(name=config.RECEIVE_STATE_SERVER, state=receive_state_server())

        fsm.add_transition(config.SETUP_STATE_SERVER, config.RECEIVE_STATE_SERVER)
        fsm.add_transition(config.RECEIVE_STATE_SERVER, config.RECEIVE_STATE_SERVER)

        self.add_behaviour(self.ServerBehav())
        self.add_behaviour(fsm)

    class ServerBehav(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's aprove it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)
            self.presence.subscribe(jid)

        async def run(self):
            self.presence.set_available()
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available


async def main():
    jid_server = "server@localhost"
    passwd_server = "server"
    server = FSMAgentServer(jid_server, passwd_server)
    await server.start()

    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await server.stop()

if __name__ == "__main__":
    spade.run(main())