from aioxmpp import PresenceShow
import logging
from spade.behaviour import OneShotBehaviour

subscription_logger = logging.getLogger("SubscriptionBehaviour")


class SubscriptionBehaviour(OneShotBehaviour):
    def __init__(self, agent_to_connect=None):
        super().__init__()
        self.agent_to_connect = agent_to_connect

    # do if agent is available
    def on_available(self, jid, stanza):
        print("[{}] Agent {} is available for receiving and sending.".format(self.agent.name, jid.split("@")[0]))
        subscription_logger.info(self.agent.name + " is available for receiving and sending")

    # do if agent is unavailable
    def on_unavailable(self, jid, stanza):
        print("[{}] Agent {} is on_unavailable for receiving and sending.".format(self.agent.name, jid.split("@")[0]))
        subscription_logger.info(self.agent.name + " is on_unavailable for receiving and sending")
        self.presence.set_available()

    # do if agent is subscribed
    def on_subscribed(self, jid):
        print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
        print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))
        subscription_logger.info(self.agent.name + " has accepted subscription")

    # do if agent subscribes
    def on_subscribe(self, jid):
        print("[{}] Agent {} asked for subscription. Let's approve it.".format(self.agent.name, jid.split("@")[0]))
        self.presence.approve(jid)
        subscription_logger.info(self.agent.name + " approved subscription")
        if "server" in self.agent.name:
            self.presence.subscribe(jid)

    # do if agent is unsubscribed
    def on_unsubscribed(self, jid):
        print("[{}] Agent {} has removed the subscription.".format(self.agent.name, jid.split("@")[0]))
        print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))
        subscription_logger.info(self.agent.name + " has removed subscription")

    # do if agent unsubscribes
    def on_unsubscribe(self, jid):
        print("[{}] Agent {} removes of subscription.".format(self.agent.name, jid.split("@")[0]))
        subscription_logger.info(jid.split("@")[0] + " removes the subscription")
        if "server" in self.agent.name:
            self.presence.unsubscribe(jid)

    async def run(self):
        # set available
        self.presence.set_available()

        # rewrite with own specified functions
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available
        self.presence.on_unavailable = self.on_unavailable
        self.presence.on_unsubscribe = self.on_unsubscribe
        self.presence.on_unsubscribed = self.on_unsubscribed

        # if agent is a client agent, subscribe to a server agent
        if "server" not in self.agent.name:
            self.presence.subscribe(self.agent_to_connect)
