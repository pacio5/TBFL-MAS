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

class SenderAgent(Agent):
    class InformBehav(OneShotBehaviour):
        async def run(self):
            print("InformBehav running")
            msg = Message(to=agent_lightwitch_username)     # Instantiate the message
            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
            msg.set_metadata("ontology", "myOntology")  # Set the ontology of the message content
            msg.set_metadata("language", "OWL-S")       # Set the language of the message content
            msg.body = "Hello World"                    # Set the message content

            await self.send(msg)
            print("Message sent!")

            # set exit_code for the behaviour
            self.exit_code = "Job Finished!"

            # stop agent from behaviour
            await self.agent.stop()

    async def setup(self):
        print("SenderAgent started")
        self.b = self.InformBehav()
        self.add_behaviour(self.b)

class ReceiverAgent(Agent):
    class RecvBehav(OneShotBehaviour):
        async def run(self):
            print("RecvBehav running")

            msg = await self.receive(timeout=10) # wait for a message for 10 seconds
            if msg:
                print("Message received with content: {}".format(msg.body))
            else:
                print("Did not received any message after 10 seconds")

            # stop agent from behaviour
            await self.agent.stop()

    async def setup(self):
        print("ReceiverAgent started")
        b = self.RecvBehav()
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b, template)


async def main():
    receiveragent = ReceiverAgent(agent_lightwitch_username, agent_lightwitch_password)
    await receiveragent.start(auto_register=True)
    print("Receiver started")

    senderagent = SenderAgent(agent_draugr_username,agent_draugr_password)
    await senderagent.start(auto_register=True)
    print("Sender started")

    await spade.wait_until_finished(receiveragent)
    print("Agents finished")


if __name__ == "__main__":
    spade.run(main())