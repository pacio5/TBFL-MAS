import logging
from spade.behaviour import OneShotBehaviour

presence_logger = logging.getLogger("PresenceBehaviour")


class PresenceBehaviour(OneShotBehaviour):
    def __init__(self, agent_to_connect=None):
        super().__init__()
        self.agent_to_connect = agent_to_connect

    def on_available(self, jid, stanza):
        print("[{}] Agent {} is available for receiving and sending.".format(self.agent.name, jid.split("@")[0]))
        presence_logger.info(self.agent.name + " is available for receiving and sending")

    def on_subscribed(self, jid):
        print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
        print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))
        presence_logger.info(self.agent.name + " has accepted subscription")

    def on_subscribe(self, jid):
        print("[{}] Agent {} asked for subscription. Let's approve it.".format(self.agent.name, jid.split("@")[0]))
        self.presence.approve(jid)
        presence_logger.info(self.agent.name + " approved subscription")
        if "server" in self.agent.name:
            self.presence.subscribe(jid)

    async def run(self):
        self.presence.set_available()
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available

        if "server" not in self.agent.name:
            self.presence.subscribe(self.agent_to_connect)
