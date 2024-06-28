import codecs
import pickle
from spade.agent import Agent
from spade.behaviour import *
import yaml
import paths
from torch import nn
from dataset import prepare_train
import torch
from models import CNN
from tqdm.contrib import tzip

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)
class set_up_state(State):
    async def run(self):
        print("-    This is the set_up state of the agent " + self.agent.name)

        '''
        If the deadtime_hard_limit is exceeded, an error message appears. 
        This limit is made up of _soft_timeout and the round_trip_time, 
        both of which are set to one minute, making the deadtime_hard_limit two minutes. 
        However, since the send took longer than two minutes, the round_trip_time was increased.
        self.agent.client.stream.round_trip_time = timedelta(minutes=25)
        '''

        # set local variables
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epoch = 0
        model = CNN(10)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_configuration"]["learning_rate"],
                                    momentum=0.9)

        # move the model to the right processor
        model.to(device)

        # save configuration variables as global variables
        self.set("criterion", criterion)
        self.set("device", device)
        self.set("epoch", epoch)
        self.set("first", True)
        self.set("model", model)
        self.set("optimizer", optimizer)
        self.set_next_state(config["client"]["receive"])


class receive_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        model = self.get("model")
        first = self.get("first")

        if first:
            print("-    This is the receive state of the agent " + self.agent.name + " at epoch " + str(epoch))

        # redo until the global model and the training data set are gotten from the server
        message = await self.receive(timeout=None)
        if message is not None:
            """messages = message.body.split(":")
            weights = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
            x_train = pickle.loads(codecs.decode(messages[1].encode(), "base64"))
            y_train = pickle.loads(codecs.decode(messages[2].encode(), "base64"))"""
            weights = pickle.loads(codecs.decode(message.body.encode(), "base64"))
            model.load_state_dict(weights)
            x_train, y_train = prepare_train()

            self.set("model", model)
            self.set("x_train", x_train)
            self.set("y_train", y_train)
            self.next_state = config["client"]["train"]
        else:
            self.set("first", False)
            self.set_next_state(config["client"]["receive"])


class train_state(State):
    async def run(self):
        # set local variables
        criterion = self.get("criterion")
        device = self.get("device")
        epoch = self.get("epoch")
        model = self.get("model")
        optimizer = self.get("optimizer")
        x_train = self.get("x_train")
        y_train = self.get("y_train")

        print("-    This is the train state of the agent " + self.agent.name + " at epoch " + str(epoch))
        time.sleep(1)

        # setting model up for training
        model.train()

        # train the local model
        for images, labels in tzip(x_train.values(), y_train.values(), desc="training local model"):
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels)
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.set_next_state(config["client"]["send"])


class send_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        model = self.get("model")
        weights = str(codecs.encode(pickle.dumps(model.state_dict()), "base64").decode())

        print("-    This is the send state of the agent " + self.agent.name + " at epoch " + str(epoch))

        try:
            message = Message(to=config["client"]["jid_server"])
            message.body = weights
            await self.send(message)
        except Exception:
            print(traceback.format_exc())
            print("sending message failed")

        # go to the next epoch if not last global epoch
        epoch += 1
        if config["learning_configuration"]["global_epochs"] > epoch:
            self.set("epoch", epoch)
            self.set_next_state(config["client"]["receive"])
