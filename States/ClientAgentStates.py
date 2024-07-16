import ast
import asyncio
import codecs
from datetime import timedelta
import logging
import pickle
from spade.behaviour import State
from spade.message import Message
import time
import torch
from torch import nn
import traceback
from Utilities.Data import Data
from Utilities.Learning import Learning
from Utilities.Metrics import Metrics
from Utilities.Models import CNN, PersonalCNN
from Utilities.Paths import config

fsm_logger = logging.getLogger("FSM")


class SetUpState(State):
    async def run(self):
        print("-    This is the set_up state of the agent " + self.agent.name)

        '''
        If the deadtime_hard_limit is exceeded, an error message appears. 
        This limit is made up of _soft_timeout and the round_trip_time, 
        both of which are set to one minute, making the deadtime_hard_limit two minutes. 
        However, since the send took longer than two minutes, the round_trip_time was increased.
        '''
        self.agent.client.stream.round_trip_time = timedelta(minutes=15)

        torch.manual_seed(self.agent.random_seed_training)

        # set local variables
        all_test_accuracies = {}
        all_test_f1_scores = {}
        all_test_precisions = {}
        all_test_recalls = {}
        all_testing_losses = {}
        all_training_losses = {}
        epoch = 1
        f1_scores_per_classes = {}
        for i in range(self.agent.args.number_of_classes_in_dataset):
            f1_scores_per_classes[str(i)] = {}
        precisions_per_classes = {}
        for i in range(self.agent.args.number_of_classes_in_dataset):
            precisions_per_classes[str(i)] = {}
        recalls_per_classes = {}
        for i in range(self.agent.args.number_of_classes_in_dataset):
            recalls_per_classes[str(i)] = {}

        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.agent.args.algorithm == "FedPER":
            model = PersonalCNN(10)
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
        optimizer = torch.optim.SGD(model.parameters(), lr=self.agent.args.learning_rate,
                                    momentum=0.9)

        # define batch sizes by iid and if the algorithm uses FedSGD or not
        batch_sizes_training_by_iid = {}
        batch_sizes_training_by_iid_and_FesSGD = {}
        batch_sizes_testing_by_iid = {}
        for i in range(self.agent.args.number_of_classes_in_dataset):
            batch_sizes_training_by_iid[str(i)] = self.agent.args.batch_size_training
            batch_sizes_training_by_iid_and_FesSGD[
                str(i)] = self.agent.args.batch_size_training * self.agent.args.local_epochs
            batch_sizes_testing_by_iid[str(i)] = self.agent.args.batch_size_testing

        # define training datasets by considering IID settings and if the algorithm uses FedSGD or not
        x_train = None
        y_train = None
        y_train_original_labels = None
        if self.agent.args.iid == 1:
            if self.agent.args.algorithm == "FesSGD":
                x_train, y_train, y_train_original_labels = Data.prepare_dataset(
                    self.agent.args.dataset_training, batch_sizes_training_by_iid_and_FesSGD,
                    1, self.agent.args.standard_deviation_for_noises,
                    self.agent.random_seed_training)
            else:
                x_train, y_train, y_train_original_labels = Data.prepare_dataset(
                    self.agent.args.dataset_training, batch_sizes_training_by_iid,
                    self.agent.args.local_epochs, self.agent.args.standard_deviation_for_noises,
                    self.agent.random_seed_training)
        elif self.agent.args.iid == 0:
            if self.agent.args.algorithm == "FesSGD":
                x_train, y_train, y_train_original_labels = Data.prepare_dataset(
                    self.agent.args.dataset_training, self.agent.batch_sizes_training_by_non_iid,
                    1, self.agent.args.standard_deviation_for_noises,
                    self.agent.random_seed_training)
            else:
                x_train, y_train, y_train_original_labels = Data.prepare_dataset(
                    self.agent.args.dataset_training, self.agent.batch_sizes_training_by_non_iid,
                    self.agent.args.local_epochs, self.agent.args.standard_deviation_for_noises,
                    self.agent.random_seed_training)
        else:
            print("wrong input for iid")

        # define testing dataset (for testing all classes are always considered)
        x_test, y_test, y_test_original_labels = Data.prepare_dataset(
            self.agent.args.dataset_testing, batch_sizes_testing_by_iid,
            self.agent.args.local_epochs, self.agent.args.standard_deviation_for_noises,
            self.agent.random_seed_testing)

        # move the model to the right processor
        model.to(device)

        # save configuration variables as global variables
        self.set("all_test_accuracies", all_test_accuracies)
        self.set("all_test_f1_scores", all_test_f1_scores)
        self.set("all_test_precisions", all_test_precisions)
        self.set("all_test_recalls", all_test_recalls)
        self.set("all_testing_losses", all_testing_losses)
        self.set("all_training_losses", all_training_losses)
        self.set("criterion", criterion)
        self.set("device", device)
        self.set("epoch", epoch)
        self.set("f1_scores_per_classes", f1_scores_per_classes)
        self.set("precisions_per_classes", precisions_per_classes)
        self.set("recalls_per_classes", recalls_per_classes)
        self.set("first", True)
        self.set("model", model)
        self.set("optimizer", optimizer)
        self.set("x_test", x_test)
        self.set("x_train", x_train)
        self.set("y_test", y_test)
        self.set("y_train", y_train)
        self.set("y_test_original_labels", y_test_original_labels)
        self.set("y_train_original_labels", y_train_original_labels)

        fsm_logger.info(self.agent.name + ": is set up")
        self.set_next_state(config["client_agent"]["receive"])


class ReceiveState(State):
    async def run(self):
        # set local variables
        epoch = self.get("epoch")
        first = self.get("first")
        model = self.get("model")
        personal_layers = self.get("personal_layers")

        if first:
            print("-    The agent " + self.agent.name + " receives a message from the central agent at epoch " + str(
                epoch))

        # redo until the weights and the epoch number of the server agent are received
        message = await self.receive(timeout=None)
        if message is not None:
            messages = message.body.split("|")
            epoch_update = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
            self.set("epoch", int(messages[1]))
            if self.agent.args.algorithm == "FedAvg" or self.agent.args.algorithm == "FedSGD":
                model.load_state_dict(epoch_update)
            elif self.agent.args.algorithm == "FedPER":
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

            self.set("model", model)

            fsm_logger.info(self.agent.name + ": received from server agent")
            self.set_next_state(config["client_agent"]["train"])
        else:
            self.set("first", False)
            self.set_next_state(config["client_agent"]["receive"])


class TrainState(State):
    async def run(self):
        # set local variables
        all_training_losses = self.get("all_training_losses")
        criterion = self.get("criterion")
        device = self.get("device")
        epoch = self.get("epoch")
        model = self.get("model")
        optimizer = self.get("optimizer")
        training_losses = []
        x_train = self.get("x_train")
        y_train = self.get("y_train")

        print("-    The agent " + self.agent.name + " trains the local model at epoch " + str(epoch))
        time.sleep(1)

        # train the model
        Learning.training(criterion, device, model, optimizer, training_losses, x_train, y_train)

        # save all_training_losses, model, training losses and optimizer
        training_loss = sum(training_losses) / len(training_losses)
        all_training_losses[str(epoch)] = training_loss
        self.set("all_training_losses", all_training_losses)
        self.set("model", model)
        self.set("training_loss", training_loss)
        self.set("optimizer", optimizer)

        fsm_logger.info(self.agent.name + ": trained local model")
        self.set_next_state(config["client_agent"]["send"])


class SendState(State):
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

        # prepare in different algorithms the weights or the gradients as epoch updates
        if self.agent.args.algorithm == "FedAvg":
            epoch_update = str(codecs.encode(pickle.dumps(model.state_dict()), "base64").decode())
        elif self.agent.args.algorithm == "FedSGD":
            epoch_update = str(codecs.encode(pickle.dumps(optimizer.state_dict()), "base64").decode())
        elif self.agent.args.algorithm == "FedPER":
            weights = model.state_dict()
            keys = []
            for key in weights.keys():
                if "pl" in key:
                    keys.append(key)
            for key in keys:
                personal_layers[key] = weights[key]
                weights.pop(key)
            epoch_update = str(codecs.encode(pickle.dumps(weights), "base64").decode())
        else:
            print("no known algorithm: " + self.agent.args.algorithm)

        # send the weights or the gradients with the losses to the server agent
        try:
            message = Message(to=self.agent.args.jid_server)
            message.body = epoch_update + "|" + epoch_loss
            await self.send(message)
        except Exception:
            print(traceback.format_exc())
            print("sending message failed")

        # save personal layers
        self.set("personal_layers", personal_layers)

        fsm_logger.info(self.agent.name + ": sent to the server")
        self.set_next_state(config["client_agent"]["predict"])


class PredictState(State):
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
        x_test = self.get("x_test")
        y_test = self.get("y_test")
        y_test_original_labels = self.get("y_test_original_labels")

        print("-    The agent " + self.agent.name + "  predicts at the epoch " + str(epoch))
        time.sleep(1)

        # predict the model
        Learning.predicting(all_labels, all_predictions, criterion, device, model, testing_losses, x_test,
                   y_test_original_labels, y_test)

        all_testing_losses[str(epoch)] = sum(testing_losses) / len(testing_losses)

        # save label and predictions as global variables
        self.set("all_labels", all_labels)
        self.set("all_predictions", all_predictions)
        self.set("all_testing_losses", all_testing_losses)

        fsm_logger.info(self.agent.name + ": predicted")
        self.set_next_state(config["client_agent"]["calculate_metrics"])


class CalculateMetricsState(State):
    async def run(self):
        # set local variables
        all_labels = self.get("all_labels")
        all_predictions = self.get("all_predictions")
        all_test_accuracies = self.get("all_test_accuracies")
        all_test_f1_scores = self.get("all_test_f1_scores")
        all_test_precisions = self.get("all_test_precisions")
        all_test_recalls = self.get("all_test_recalls")
        epoch = self.get("epoch")
        f1_scores_per_classes = self.get("f1_scores_per_classes")
        precisions_per_classes = self.get("precisions_per_classes")
        recalls_per_classes = self.get("recalls_per_classes")

        print(
            "-    The agent " + self.agent.name +
            "  calculates the accuracy, f1-score, precision and recall at the epoch "
            + str(epoch))

        # calculate metrics,
        Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                          all_test_precisions, all_test_recalls, epoch)
        Metrics.calculate_f1_score_per_classes(all_labels, all_predictions, epoch, f1_scores_per_classes)
        Metrics.calculate_precisions_per_classes(all_labels, all_predictions, epoch, precisions_per_classes)
        Metrics.calculate_recalls_per_classes(all_labels, all_predictions, epoch, recalls_per_classes)

        # save metrics as global variables
        self.set("all_test_accuracies", all_test_accuracies)
        self.set("all_test_f1_scores", all_test_f1_scores)
        self.set("all_test_precisions", all_test_precisions)
        self.set("all_test_recalls", all_test_recalls)
        self.set("f1_scores_per_classes", f1_scores_per_classes)
        self.set("precisions_per_classes", precisions_per_classes)
        self.set("recalls_per_classes", recalls_per_classes)

        fsm_logger.info(self.agent.name + ": metrics calculated")

        # control if the agent has left the MAS or not
        leave = False
        await asyncio.sleep(30)
        for agent in list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items())):
            if agent[1]["subscription"] != "both" and "server" in str(agent[1]["presence"]):
                leave = True

        '''
            If it's not the last global epoch and the agent didn't leave the MAS,
            then proceed to the next one. 
            Otherwise, store the collected metrics.
        '''
        if self.agent.args.global_epochs > epoch and not leave:
            self.set("epoch", epoch + 1)
            self.set_next_state(config["client_agent"]["receive"])
        else:
            self.set_next_state(config["client_agent"]["store_metrics"])


class StoreMetricsState(State):
    async def run(self):
        # set local variables
        all_test_accuracies = self.get("all_test_accuracies")
        all_test_f1_scores = self.get("all_test_f1_scores")
        all_testing_losses = self.get("all_testing_losses")
        all_test_precisions = self.get("all_test_precisions")
        all_test_recalls = self.get("all_test_recalls")
        all_training_losses = self.get("all_training_losses")
        epoch = self.get("epoch")
        f1_scores_per_classes = self.get("f1_scores_per_classes")
        precisions_per_classes = self.get("precisions_per_classes")
        recalls_per_classes = self.get("recalls_per_classes")

        print("-    The agent " + self.agent.name + "  stores the metrics at the epoch " + str(epoch))

        # show last epoch if the threshold is reached
        if self.agent.args.wait_until_threshold_is_reached != "no threshold":
            print("Epochs needed to reach the threshold " + str(self.agent.args.threshold) + ": " + str(epoch))

        # store metrics
        Metrics.store_metrics(self.agent.name, all_test_accuracies, all_test_f1_scores, all_test_precisions, all_test_recalls,
                      all_testing_losses, all_training_losses, self.agent.args,
                      self.agent.batch_sizes_training_by_non_iid, f1_scores_per_classes, precisions_per_classes,
                      recalls_per_classes)

        fsm_logger.info(self.agent.name + ": metrics stored")
