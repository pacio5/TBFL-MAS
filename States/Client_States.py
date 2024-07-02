import ast
import asyncio
import codecs
from dataset import prepare_dataset
from learning import training
import logging
from models import CNN, Personal
import paths
import pickle
from spade.behaviour import State
from spade.message import Message
import time
import torch
from torch import nn
from tqdm.contrib import tzip
import traceback
import yaml

fsm_logger = logging.getLogger("FSM")

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
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epoch = 0
        if config["learning_configuration"]["FL"]["algorithm"] == "p-Fed":
            model = Personal(10)
            personal_layers = {}
            weights = model.state_dict()
            keys = []
            for key in weights.keys():
                if "pl" in key:
                    keys.append(key)
            for key in keys:
                personal_layers[key] = weights[key]
            self.set("personal_layers", personal_layers)
        else:
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
        fsm_logger.info(self.agent.name + ": is set up")
        self.set_next_state(config["client"]["receive"])


class receive_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        first = self.get("first")
        model = self.get("model")
        personal_layers = self.get("personal_layers")

        if first:
            print("-    The agent " + self.agent.name + " receives a message from the central agent at epoch " + str(epoch))

        # redo until the global model and the training data set are gotten from the server
        message = await self.receive(timeout=None)
        if message is not None:
            """messages = message.body.split("|")
            weights = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
            x_train = pickle.loads(codecs.decode(messages[1].encode(), "base64"))
            y_train = pickle.loads(codecs.decode(messages[2].encode(), "base64"))"""
            if message.body == "finish":
                self.set_next_state(config["client"]["finish"])
            else:
                messages = message.body.split("|")
                epoch_update = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
                batch_size = int(messages[1])
                classes_of_data_object = ast.literal_eval(messages[2])
                if config["learning_configuration"]["FL"]["algorithm"] == "FedAvg" or config["learning_configuration"]["FL"]["algorithm"] == "FedSGD":
                    model.load_state_dict(epoch_update)
                elif config["learning_configuration"]["FL"]["algorithm"] == "p-Fed":
                    # pop last two keys: weight and bias
                    last_layers = {}
                    last_weight_Key = list(epoch_update.keys())[-2]
                    last_layers[last_weight_Key] = epoch_update.pop(last_weight_Key)
                    last_bias_Key = list(epoch_update.keys())[-1]
                    last_layers[last_bias_Key] = epoch_update.pop(last_bias_Key)
                    for personal_layer in personal_layers:
                        epoch_update[personal_layer] = personal_layers[personal_layer]
                    for key in last_layers.keys():
                        epoch_update[key] = last_layers[key]
                    model.load_state_dict(epoch_update)
                else:
                    print("not configured algorithm")

                self.set("batch_size", batch_size)
                self.set("model", model)
                self.set("classes_of_data_object", classes_of_data_object)

                fsm_logger.info(self.agent.name + ": received from server agent")
                self.next_state = config["client"]["train"]
        else:
            self.set("first", False)
            self.set_next_state(config["client"]["receive"])


class train_state(State):
    async def run(self):
        # set local variables
        batch_size = self.get("batch_size")
        criterion = self.get("criterion")
        device = self.get("device")
        epoch = self.get("epoch")
        model = self.get("model")
        optimizer = self.get("optimizer")
        classes_of_data_object = self.get("classes_of_data_object")
        training_losses = []

        x_train, y_train, y_original_labels = prepare_dataset(
            config["learning_configuration"]["dataset_training"],
            classes_of_data_object, batch_size, config["learning_configuration"]["local_epochs"])


        print("-    The agent " + self.agent.name + " trains the local model at epoch " + str(epoch))
        time.sleep(1)

        training(criterion, device, model, optimizer, training_losses, x_train, y_train)

        # save model, training losses and optimizer
        self.set("model", model)
        self.set("training_loss", sum(training_losses)/len(training_losses))
        self.set("optimizer", optimizer)

        fsm_logger.info(self.agent.name + ": trained local model")
        self.set_next_state(config["client"]["send"])




class send_state(State):
    async def run(self):
        # set local variables
        training_loss = self.get("training_loss")
        epoch = self.get("epoch")
        epoch_loss = str(training_loss)
        epoch_update = ""
        model = self.get("model")
        optimizer = self.get("optimizer")
        personal_layers = {}


        print("-    The agent " + self.agent.name + " sends a message to the central agent at epoch " + str(epoch))

        if config["learning_configuration"]["FL"]["algorithm"] == "FedAvg":
            epoch_update = str(codecs.encode(pickle.dumps(model.state_dict()), "base64").decode())
        elif config["learning_configuration"]["FL"]["algorithm"] == "FedSGD":
            epoch_update = str(codecs.encode(pickle.dumps(optimizer.state_dict()), "base64").decode())
        elif config["learning_configuration"]["FL"]["algorithm"] == "p-Fed":
            weights = model.state_dict()
            keys = []
            for key in weights.keys():
                if "pl" in key:
                    keys.append(key)
            for key in keys:
                personal_layers[key] = weights[key]
                weights.pop(key)
            epoch_update = str(codecs.encode(pickle.dumps(weights), "base64").decode())

        try:
            message = Message(to=config["client"]["jid_server"])
            message.body = epoch_update + "|" + epoch_loss
            await self.send(message)
        except Exception:
            print(traceback.format_exc())
            print("sending message failed")

        # save personal layers
        self.set("personal_layers", personal_layers)

        epoch += 1
        self.set("epoch", epoch)
        fsm_logger.info(self.agent.name + ": sent to the server")
        self.set_next_state(config["client"]["receive"])

class finish_state(State):
    async def run(self):
        print("-    This is the finish state of the agent " + self.agent.name)
        fsm_logger.info(self.agent.name + ": finished")
