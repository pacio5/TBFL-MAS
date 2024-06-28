from spade.agent import Agent
from spade.behaviour import *
import paths
import uuid
import yaml

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)
class ExtendedFSMBehaviour(FSMBehaviour):
    async def on_start(self):
        print(f"{self.agent.name} starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"{self.agent.name} finished at state {self.current_state}")
        self.agent.presence.set_unavailable()
        await self.agent.stop()

class PresenceBehaviour(OneShotBehaviour):
    def __init__(self, agent_to_connect=None):
        super().__init__()
        self.agent_to_connect = agent_to_connect

    def on_available(self, jid, stanza):
        print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

    def on_subscribed(self, jid):
        print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
        print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

    def on_subscribe(self, jid):
        print("[{}] Agent {} asked for subscription. Let's approve it.".format(self.agent.name, jid.split("@")[0]))
        self.presence.approve(jid)
        if "server" in self.agent.name:
            self.presence.subscribe(jid)

    async def run(self):
        self.presence.set_available()
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available

        if "server" not in self.agent.name:
            self.presence.subscribe(self.agent_to_connect)
