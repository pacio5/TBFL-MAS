from spade.agent import Agent
from spade.behaviour import *
from spade.message import *
import spade
import os
from dotenv import load_dotenv

load_dotenv()

agent_lightwitch_username = os.environ.get('agent_lightwitch_username')
agent_lightwitch_password = os.environ.get('agent_lightwitch_password')
agent_draugr_username = os.environ.get('agent_draugr_username')
agent_draugr_password = os.environ.get('agent_draugr_password')

class Server(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(self.ServerBehav())

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
    server = Server(jid_server, passwd_server)
    await server.start()

    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await server.stop()

if __name__ == "__main__":
    spade.run(main())