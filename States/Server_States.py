import codecs
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from tqdm import tqdm
from tqdm.contrib import tzip
from models import CNN
from dataset import prepare_train, prepare_test
from spade.agent import Agent
from spade.behaviour import *
import spade
import yaml
import paths
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)

class set_up_state(State):
    async def run(self):
        print("-    This is the set_up state of the central agent")

        '''
        If the deadtime_hard_limit is exceeded, an error message appears. 
        This limit is made up of _soft_timeout and the round_trip_time, 
        both of which are set to one minute, making the deadtime_hard_limit two minutes. 
        However, since the send took longer than two minutes, the round_trip_time was increased.
        self.agent.client.stream.round_trip_time = timedelta(minutes=10)
        '''

        # set local variables
        all_accuracies = []
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epoch = 0
        model = CNN(10)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_configuration"]["learning_rate"],
                                    momentum=0.9)

        # get test data set for the central server
        x_test, y_test = prepare_test()

        # move the model to the right processor
        model.to(device)

        # save configuration variables as global variables
        self.set("all_accuracies", all_accuracies)
        self.set("all_f1_scores", all_f1_scores)
        self.set("all_precisions", all_precisions)
        self.set("all_recalls", all_recalls)
        self.set("criterion", criterion)
        self.set("device", device)
        self.set("epoch", epoch)
        self.set("model", model)
        self.set("optimizer", optimizer)
        self.set("x_test", x_test)
        self.set("y_test", y_test)

        '''
        If FL then sending is the next state. 
        Otherwise prepare training data set for ML and save it as global variables and training is the next state.
        '''
        if config["learning_configuration"]["FL"]:
            self.set_next_state(config["server"]["control_agents_present"])
        else:
            x_train, y_train = prepare_train()
            self.set("x_train", x_train)
            self.set("y_train", y_train)
            self.set_next_state(config["server"]["train"])


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

        print("-    This is the train state of the central agent at the epoch " + str(epoch))
        time.sleep(1)

        # setting model up for training
        model.train()

        # train the global model
        for images, labels in tzip(x_train[epoch], y_train[epoch]):
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels)
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.set_next_state(config["server"]["predict"])

class control_agents_present_state(State):
    async def run(self):
        await asyncio.sleep(30)
        if config["learning_configuration"]["number_of_clients"] > len(list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))):
            self.set_next_state(config["server"]["control_agents_present"])
        else:
            self.set_next_state(config["server"]["send"])

class send_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        model = self.get("model")
        weights = str(codecs.encode(pickle.dumps(model.state_dict()), "base64").decode())

        print("-    This is the send state of the central agent at the epoch " + str(epoch))
        time.sleep(1)

        clients = list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))
        time.sleep(1)

        # send global weights and training data set to all present registered clients
        pbar = tqdm(total=len(clients), desc="sending weights to all present registered clients")
        message_sended = clients.copy()
        while len(message_sended) > 0:
            try:
                client = message_sended[0]
                """x_train, y_train = prepare_train()
                x_train_data = str(codecs.encode(pickle.dumps(x_train), "base64").decode())
                y_train_data = str(codecs.encode(pickle.dumps(y_train), "base64").decode())
                message.body = weights + ":" + x_train_data + ":" + y_train_data"""
                message = Message(to=str(client[0]))
                message.body = weights
                await self.send(message)
                message_sended.pop(0)
                pbar.update(1)
            except Exception:
                print(traceback.format_exc())
                print("sending message failed")
        pbar.close()

        self.set_next_state(config["server"]["receive"])


class receive_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        weights = {}

        print("-    This is the receive state of the central agent at the epoch " + str(epoch))
        await asyncio.sleep(config["learning_configuration"]["number_of_clients"]*15)

        # receive weights from the clients
        pbar = tqdm(total=config["learning_configuration"]["number_of_clients"], desc="receive weights from the clients")
        while len(weights) < config["learning_configuration"]["number_of_clients"]:
            message = await self.receive(timeout=None)
            if message is not None:
                weights[message.sender] = pickle.loads(codecs.decode(message.body.encode(), "base64"))
                pbar.update(1)
        pbar.close()

        # save weights as global variable
        self.set("weights", weights)

        self.set_next_state(config["server"]["avg"])


class avg_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        model = self.get("model")
        weights = self.get("weights")
        avg = {}
        firstKey = list(weights.keys())[0]
        avg["avg"] = weights[firstKey]

        print("-    This is the avg state of the central agent at the epoch " + str(epoch))
        time.sleep(1)

        # average weights
        for i in tqdm(weights.keys()):
            for j in tqdm(weights[i].keys()):
                avg["avg"][j].add(weights[i][j])

        for i in tqdm(avg["avg"].keys()):
            torch.mean(avg["avg"][i], dtype=torch.float64)

        # load and save averaged weights in the global model
        model.load_state_dict(avg["avg"])
        self.set("model", model)

        self.set_next_state(config["server"]["predict"])


class predict_state(State):
    async def run(self):
        # set local variables
        all_labels = []
        all_predictions = []
        device = self.get("device")
        epoch = self.get("epoch")
        model = self.get("model")
        x_test = self.get("x_test")
        y_test = self.get("y_test")

        print("-    This is the predict state of the central agent at the epoch " + str(epoch))
        time.sleep(1)

        # setting model up for evaluating
        model.eval()

        # calculate predictions
        with torch.no_grad():
            for images, labels in tzip(x_test[epoch], y_test[epoch]):
                images = torch.from_numpy(images)
                labels = torch.from_numpy(labels)
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        # save label and predictions as global variables
        self.set("all_labels", all_labels)
        self.set("all_predictions", all_predictions)

        self.set_next_state(config["server"]["collect_metrics"])


class collect_metrics_state(State):
    async def run(self):
        # set local variables
        all_accuracies = self.get("all_accuracies")
        all_f1_scores = self.get("all_f1_scores")
        all_labels = self.get("all_labels")
        all_precisions = self.get("all_precisions")
        all_predictions = self.get("all_predictions")
        all_recalls = self.get("all_recalls")
        epoch = self.get("epoch")

        print("-    This is the collect metrics state of the central agent at the epoch " + str(epoch))

        # calculate metrics
        all_accuracies.append(accuracy_score(all_labels, all_predictions))
        all_f1_scores.append(
            f1_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))
        all_precisions.append(
            precision_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))
        all_recalls.append(
            recall_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))

        # save metrics as global variables
        self.set("all_accuracies", all_accuracies)
        self.set("all_f1_scores", all_f1_scores)
        self.set("all_precisions", all_precisions)
        self.set("all_recalls", all_recalls)

        '''
        If it's not the last global epoch, proceed to the next one. 
        Otherwise, plot the collected metrics. 
        For FL, the next epoch starts by sending global weights.
        However, with ML it starts with training the global model.
        '''
        epoch += 1
        if config["learning_configuration"]["global_epochs"] > epoch:
            self.set("epoch", epoch)
            if config["learning_configuration"]["FL"]:
                self.set_next_state(config["server"]["send"])
            else:
                self.set_next_state(config["server"]["train"])

        else:
            self.set_next_state(config["server"]["plot_metrics"])


class plot_metrics_state(State):
    async def run(self):
        # set local variables
        all_accuracies = self.get("all_accuracies")
        all_f1_scores = self.get("all_f1_scores")
        all_precisions = self.get("all_precisions")
        all_recalls = self.get("all_recalls")
        epoch = self.get("epoch")

        print("-    This is the plot metrics state of the central agent at the epoch " + str(epoch))

        # show metrics in a plot
        df = pd.DataFrame(
            {'accuracy': all_accuracies, 'precision': all_precisions, 'recall': all_recalls, 'f1': all_f1_scores},
            index=np.arange(len(all_accuracies)))
        df.plot(title="Metrics Plot")
        plt.show()
        #labelling the plot