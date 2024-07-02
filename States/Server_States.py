import asyncio
import ast
import codecs
from dataset import prepare_dataset
from learning import average_gradients, average_weights, calculate_metrics, predicting, training
import logging
from matplotlib import pyplot as plt
from models import CNN, Personal
import numpy as np
import pandas as pd
import paths
import pickle
from spade.behaviour import State
from spade.message import Message
import time
import torch
from torch import nn
from tqdm import tqdm
import traceback
import yaml

fsm_logger = logging.getLogger("FSM")

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
        all_test_accuracies = []
        all_test_f1_scores = []
        all_testing_losses = []
        all_test_precisions = []
        all_test_recalls = []
        all_training_losses = []
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epoch = 0
        model = CNN(10)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_configuration"]["learning_rate"],
                                    momentum=0.9)

        # move the model to the right processor
        model.to(device)

        # save configuration variables as global variables
        self.set("all_test_accuracies", all_test_accuracies)
        self.set("all_test_f1_scores", all_test_f1_scores)
        self.set("all_testing_losses", all_testing_losses)
        self.set("all_test_precisions", all_test_precisions)
        self.set("all_test_recalls", all_test_recalls)
        self.set("all_training_losses", all_training_losses)
        self.set("criterion", criterion)
        self.set("device", device)
        self.set("epoch", epoch)
        self.set("model", model)
        self.set("optimizer", optimizer)

        fsm_logger.info(self.agent.name + ": is set up")
        '''
        If FL then sending is the next state. 
        Otherwise prepare training data set for ML and save it as global variables and training is the next state.
        '''
        if config["learning_configuration"]["FL_or_ML"] == "FL":
            classes_of_data_object = {}
            for i in range(config["learning_configuration"]["FL"]["number_of_agents"]):
                classes_of_data_object[str(i)] = list(np.random.choice(10, config["learning_configuration"]["FL"]["number_of_classes_per_client"][i]))
            self.set("classes_of_data_object", classes_of_data_object)
            self.set_next_state(config["server"]["control_agents_present"])
        else:
            self.set_next_state(config["server"]["train"])


class train_state(State):
    async def run(self):
        # set local variables
        all_training_losses = self.get("all_training_losses")
        criterion = self.get("criterion")
        device = self.get("device")
        epoch = self.get("epoch")
        model = self.get("model")
        optimizer = self.get("optimizer")
        training_losses = []
        x_train, y_train, y_original_labels = prepare_dataset(
                config["learning_configuration"]["dataset_training"],
                batch_size=config["learning_configuration"]["ML"]["batch_size_training"],
                local_epochs=config["learning_configuration"]["local_epochs"])

        print("-    The central agent trains the global model at the epoch " + str(epoch))
        time.sleep(1)

        training(criterion, device, model, optimizer, training_losses, x_train, y_train)

        # save training losses as global variable
        all_training_losses.append(sum(training_losses) / len(training_losses))
        self.set("all_training_losses", all_training_losses)

        fsm_logger.info(self.agent.name + ": trained global model")
        self.set_next_state(config["server"]["predict"])

class control_agents_present_state(State):
    async def run(self):
        await asyncio.sleep(30)
        if config["learning_configuration"]["FL"]["number_of_agents"] > len(list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))):
            self.set_next_state(config["server"]["control_agents_present"])
        else:
            fsm_logger.info(self.agent.name + ": agents are present")
            self.set_next_state(config["server"]["send"])

class send_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        model = self.get("model")
        classes_of_data_object = self.get("classes_of_data_object")
        weights = str(codecs.encode(pickle.dumps(model.state_dict()), "base64").decode())

        print("-    The central agent sends messages to the client agents at the epoch " + str(epoch))
        time.sleep(1)

        agents = list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))
        time.sleep(1)

        # send global weights and training data set to all present registered agents
        pbar = tqdm(total=len(list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))),
                    desc="sending to all present registered agents")
        message_sended = agents.copy()
        while len(message_sended) > 0:
            try:
                client = message_sended[0]
                """x_train, y_train = prepare_train()
                x_train_data = str(codecs.encode(pickle.dumps(x_train), "base64").decode())
                y_train_data = str(codecs.encode(pickle.dumps(y_train), "base64").decode())
                message.body = weights + "|" + x_train_data + "|" + y_train_data"""
                message = Message(to=str(client[0]))
                message.body = weights + "|" + str(config["learning_configuration"]["FL"]["batch_size_training_per_client"][pbar.n]) + "|" + str(classes_of_data_object[str(pbar.n)])
                await self.send(message)
                message_sended.pop(0)
                pbar.update(1)
            except Exception:
                print(traceback.format_exc())
                print("sending message failed")
        pbar.close()

        fsm_logger.info(self.agent.name + ": sent to agents")
        self.set_next_state(config["server"]["receive"])


class receive_state(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        epoch_updates = {}
        losses = {}

        print("-    The central agent receives messages from the client agents at the epoch " + str(epoch))
        await asyncio.sleep(config["learning_configuration"]["FL"]["number_of_agents"]*15)

        # receive weights from the agents
        pbar = tqdm(total=len(list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))),
                    desc="receive epoch updates and losses from the agents")
        while len(epoch_updates) < len(list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))):
            message = await self.receive(timeout=None)
            if message is not None:
                messages = message.body.split("|")
                if config["learning_configuration"]["FL"]["algorithm"] == "FedSGD":
                    epoch_updates[message.sender] = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
                else:
                    epoch_updates[message.sender] = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
                losses[message.sender] = float(messages[1])
                pbar.update(1)
        pbar.close()

        # save epoch_updates and losses as global variable
        self.set("epoch_updates", epoch_updates)
        self.set("losses", losses)

        fsm_logger.info(self.agent.name + ": received from agents")
        self.set_next_state(config["server"]["avg"])


class avg_state(State):
    async def run(self):
        # set local variables
        all_training_losses = self.get("all_training_losses")
        epoch = self.get("epoch")
        epoch_updates = self.get("epoch_updates")
        losses = self.get("losses")
        model = self.get("model")
        optimizer = self.get("optimizer")
        avg = {}
        print("-    The central agent averages at the epoch " + str(epoch))
        time.sleep(1)

        if config["learning_configuration"]["FL"]["algorithm"] == "FedSGD":
            # average gradients and make gradient descent step
            average_gradients(avg, epoch_updates)
            optimizer.load_state_dict(avg["gradients"])
            optimizer.step()
        else:
            # average weights and load to model
            average_weights(avg, epoch_updates)
            model.load_state_dict(avg["weights"])

        # average losses and save them globally
        all_training_losses.append(sum(losses.values()) / len(losses))
        self.set("all_training_losses", all_training_losses)

        # save updated global model
        self.set("model", model)

        fsm_logger.info(self.agent.name + ": averaged")
        self.set_next_state(config["server"]["predict"])



class predict_state(State):
    async def run(self):
        # set local variables
        all_labels = []
        all_predictions = []
        all_testing_losses = self.get("all_testing_losses")
        criterion = self.get("criterion")
        device = self.get("device")
        epoch = self.get("epoch")
        model = self.get("model")
        testing_losses = []
        x_test, y_test, y_original_labels = prepare_dataset(
                config["learning_configuration"]["dataset_testing"],
                batch_size=config["learning_configuration"]["batch_size_testing"],
                local_epochs=config["learning_configuration"]["local_epochs"])

        print("-    The central agent predicts at the epoch " + str(epoch))
        time.sleep(1)

        predicting(all_labels, all_predictions, criterion, device, model, testing_losses, x_test,
                           y_original_labels, y_test)

        all_testing_losses.append(sum(testing_losses)/len(testing_losses))

        # save label and predictions as global variables
        self.set("all_labels", all_labels)
        self.set("all_predictions", all_predictions)
        self.set("all_testing_losses", all_testing_losses)

        fsm_logger.info(self.agent.name + ": predicted")
        self.set_next_state(config["server"]["collect_metrics"])


class collect_metrics_state(State):
    async def run(self):
        # set local variables
        all_test_accuracies = self.get("all_test_accuracies")
        all_test_f1_scores = self.get("all_test_f1_scores")
        all_labels = self.get("all_labels")
        all_test_precisions = self.get("all_test_precisions")
        all_predictions = self.get("all_predictions")
        all_test_recalls = self.get("all_test_recalls")
        epoch = self.get("epoch")

        print("-    The central agent calculates the accuracy, f1-score, precision and recall at the epoch " + str(epoch))

        # calculate metrics
        calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                     all_test_precisions, all_test_recalls)

        # save metrics as global variables
        self.set("all_test_accuracies", all_test_accuracies)
        self.set("all_test_f1_scores", all_test_f1_scores)
        self.set("all_test_precisions", all_test_precisions)
        self.set("all_test_recalls", all_test_recalls)

        '''
        If it's not the last global epoch, proceed to the next one. 
        Otherwise, plot the collected metrics. 
        For FL, the next epoch starts by sending global weights.
        However, with ML it starts with training the global model.
        '''
        epoch += 1
        if config["learning_configuration"]["wait_until_accuracy_is_reached"]:
            print(all_test_accuracies[-1])
            if all_test_accuracies[-1] > config["learning_configuration"]["accuracy_treshold"]:
                fsm_logger.info(self.agent.name + ": metrics collected")
                self.set_next_state(config["server"]["plot_metrics"])
            else:
                self.set("epoch", epoch)
                if config["learning_configuration"]["FL_or_ML"] == "FL":
                    fsm_logger.info(self.agent.name + ": metrics collected")
                    self.set_next_state(config["server"]["send"])
                else:
                    fsm_logger.info(self.agent.name + ": metrics collected")
                    self.set_next_state(config["server"]["train"])

        else:
            if config["learning_configuration"]["global_epochs"] > epoch:
                self.set("epoch", epoch)
                if config["learning_configuration"]["FL_or_ML"] == "FL":
                    fsm_logger.info(self.agent.name + ": metrics collected")
                    self.set_next_state(config["server"]["send"])
                else:
                    fsm_logger.info(self.agent.name + ": metrics collected")
                    self.set_next_state(config["server"]["train"])
            else:
                fsm_logger.info(self.agent.name + ": metrics collected")
                self.set_next_state(config["server"]["plot_metrics"])



class plot_metrics_state(State):
    async def run(self):
        # set local variables
        all_test_accuracies = self.get("all_test_accuracies")
        all_test_f1_scores = self.get("all_test_f1_scores")
        all_testing_losses = self.get("all_testing_losses")
        all_test_precisions = self.get("all_test_precisions")
        all_test_recalls = self.get("all_test_recalls")
        all_training_losses = self.get("all_training_losses")
        epoch = self.get("epoch")

        print("-    The central agent plots the metrics at the epoch " + str(epoch))

        if config["learning_configuration"]["wait_until_accuracy_is_reached"]:
            print("Epochs needed to reach the treshold " + str(config["learning_configuration"]["accuracy_treshold"]) + ": " + str(epoch))

        # show metrics in a plot
        df = pd.DataFrame(
            {'test accuracy': all_test_accuracies, 'test precision': all_test_precisions, 'test recall': all_test_recalls, 'test f1': all_test_f1_scores},
            index=np.arange(len(all_test_accuracies)))
        df.plot(title="performance metrics", xlabel="global epochs", ylabel="percentage")
        plt.show()

        df = pd.DataFrame(
            {'test losses': all_testing_losses, 'training losses': all_training_losses},
            index=np.arange(len(all_testing_losses)))
        df.plot(title="losses", xlabel="global epochs", ylabel="percentage losses")
        plt.show()
        fsm_logger.info(self.agent.name + ": metrics plotted")
        self.set_next_state(config["server"]["finish"])

class finish_state(State):
    async def run(self):
        print("-    This is the finish state of the central agent")

        if config["learning_configuration"]["FL_or_ML"] == "FL":
            agents = list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items()))[
                      :config["learning_configuration"]["FL"]["number_of_agents"]]
            time.sleep(1)

            # send global weights and training data set to all present registered agents
            pbar = tqdm(total=len(agents), desc="sending to all present registered agents")
            message_sended = agents.copy()
            while len(message_sended) > 0:
                try:
                    client = message_sended[0]
                    message = Message(to=str(client[0]))
                    message.body = "finish"
                    await self.send(message)
                    message_sended.pop(0)
                    pbar.update(1)
                except Exception:
                    print(traceback.format_exc())
                    print("sending message failed")
            pbar.close()
        fsm_logger.info(self.agent.name + ": finished")

