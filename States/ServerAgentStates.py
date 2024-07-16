import asyncio
import ast
import codecs
from datetime import timedelta
import logging
import pickle
from spade.behaviour import State
from spade.message import Message
import time
import torch
from torch import nn
from tqdm import tqdm
from Utilities.Data import Data
from Utilities.Learning import Learning
from Utilities.Metrics import Metrics
from Utilities.Models import CNN
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
        model = CNN(10)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.agent.args.learning_rate,
                                    momentum=0.9)

        # define batch sizes by iid
        batch_sizes_training_by_iid = {}
        batch_sizes_testing_by_iid = {}
        for i in range(self.agent.args.number_of_classes_in_dataset):
            batch_sizes_training_by_iid[str(i)] = self.agent.args.batch_size_training
            batch_sizes_testing_by_iid[str(i)] = self.agent.args.batch_size_testing

        # define training datasets by considering IID settings
        x_train = None
        y_train = None
        y_train_original_labels = None
        if self.agent.args.iid == 1:
            x_train, y_train, y_train_original_labels = Data.prepare_dataset(
                self.agent.args.dataset_training, batch_sizes_training_by_iid,
                self.agent.args.local_epochs, self.agent.args.standard_deviation_for_noises,
                self.agent.random_seed_training)
        elif self.agent.args.iid == 0:
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
        self.set("f1_scores_per_classes", f1_scores_per_classes)
        self.set("precisions_per_classes", precisions_per_classes)
        self.set("recalls_per_classes", recalls_per_classes)
        self.set("model", model)
        self.set("optimizer", optimizer)
        self.set("x_test", x_test)
        self.set("x_train", x_train)
        self.set("y_test", y_test)
        self.set("y_train", y_train)
        self.set("y_test_original_labels", y_test_original_labels)
        self.set("y_train_original_labels", y_train_original_labels)

        fsm_logger.info(self.agent.name + ": is set up")

        # When ML then training is the next state, otherwise control agents present is the next state.
        if self.agent.args.algorithm == "ML":
            self.set_next_state(config["server"]["train"])
        else:
            await asyncio.sleep(20 * self.agent.args.number_of_client_agents)
            self.set_next_state(config["server"]["control_agents_present"])


class TrainState(State):
    async def run(self):
        # set local variables
        all_training_losses = self.get("all_training_losses")
        criterion = self.get("criterion")
        device = self.get("device")
        model = self.get("model")
        optimizer = self.get("optimizer")
        training_losses = []
        x_train = self.get("x_train")
        y_train = self.get("y_train")

        print("-    The agent " + self.agent.name + "  trains the global model at the epoch " +
              str(self.agent.args.epoch))
        time.sleep(1)

        # train the model
        Learning.training(criterion, device, model, optimizer, training_losses, x_train, y_train)

        # save training losses as global variable
        all_training_losses[str(self.agent.args.epoch)] = sum(training_losses) / len(training_losses)
        self.set("all_training_losses", all_training_losses)

        fsm_logger.info(self.agent.name + ": trained global model")
        self.set_next_state(config["server"]["predict"])


class ControlAgentsPresentState(State):
    async def run(self):
        await asyncio.sleep(30)

        # list the subscribed agents that are available
        agentsAvaible = []
        agentsSubscribed = []
        for agent in list(filter(lambda x: ("presence" in x[1]), self.agent.presence.get_contacts().items())):
            if agent[1]["subscription"] == "both":
                agentsSubscribed.append(agent)
                if "type=<PresenceType.AVAILABLE: None>>" in str(agent[1]["presence"]):
                    agentsAvaible.append(agent)

        # control of all subscribed agents are available
        if (len(agentsAvaible) == len(agentsSubscribed)) and (len(agentsSubscribed) >= self.agent.args.number_of_client_agents-1):
            self.set("agents", agentsSubscribed)
            fsm_logger.info(self.agent.name + ": client agents are present")
            self.set_next_state(config["server"]["send"])
        else:
            self.set_next_state(config["server"]["control_agents_present"])



class SendState(State):
    async def run(self):
        # set local variables
        agents = self.get("agents")
        model = self.get("model")
        weights = str(codecs.encode(pickle.dumps(model.state_dict()), "base64").decode())

        print("-    The agent " + self.agent.name + "  sends messages to the client agents at the epoch " +
              str(self.agent.args.epoch))
        time.sleep(1)

        # send global weights and training data set to all present registered agents
        pbar = tqdm(total=len(agents),
                    desc="sending to all present registered agents")
        agents_sent = []
        while len(agents) > 0:
            try:
                agent = agents[0]
                message = Message(to=str(agent[0]))
                message.body = weights + "|" + str(self.agent.args.epoch)
                await self.send(message)
                agents_sent.append(agents.pop(0))
                pbar.update(1)
            except Exception:
                failed_agent = agents.pop(0)
                print("sending message to " + str(failed_agent[0]) + " failed")
        pbar.close()

        self.set("agents", agents_sent)
        fsm_logger.info(self.agent.name + ": sent to client agents")
        self.set_next_state(config["server"]["receive"])


class ReceiveState(State):
    async def run(self):
        # set local variables
        agents = self.get("agents")
        epoch_updates = {}
        losses = {}

        print("-    The agent " + self.agent.name + "  receives messages from the client agents at the epoch " + str(
            self.agent.args.epoch))
        await asyncio.sleep(self.agent.args.number_of_client_agents * 40)

        # receive weights from the agents
        pbar = tqdm(total=len(agents),
                    desc="receive epoch updates and losses from the agents")
        while len(epoch_updates) < len(agents):
            message = await self.receive(timeout=None)
            if message is not None:
                messages = message.body.split("|")
                if self.agent.args.algorithm == "FedSGD":
                    epoch_updates[message.sender] = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
                else:
                    epoch_updates[message.sender] = pickle.loads(codecs.decode(messages[0].encode(), "base64"))
                losses[message.sender] = float(messages[1])
                pbar.update(1)
            if pbar.format_dict["elapsed"] > 500:
                break
        pbar.close()
        # save epoch_updates and losses as global variable
        self.set("epoch_updates", epoch_updates)
        self.set("losses", losses)

        fsm_logger.info(self.agent.name + ": received from client agents")
        self.set_next_state(config["server"]["avg"])


class AvgState(State):
    async def run(self):
        # set local variables
        all_training_losses = self.get("all_training_losses")
        criterion = self.get("criterion")
        device = self.get("device")
        epoch_updates = self.get("epoch_updates")
        losses = self.get("losses")
        model = self.get("model")
        optimizer = self.get("optimizer")
        avg = {}
        x_train = self.get("x_train")
        y_train = self.get("y_train")

        print("-    The agent " + self.agent.name + "  averages at the epoch " + str(self.agent.args.epoch))
        time.sleep(1)

        if self.agent.args.algorithm == "FedSGD":
            # average gradients and make gradient descent step
            Learning.average_gradients(avg, epoch_updates)
            optimizer.load_state_dict(avg["gradients"])
            Learning.gradient_descent(criterion, device, model, optimizer, x_train, y_train)
            self.set("optimizer", optimizer)
        else:
            # average weights and load to model
            Learning.average_weights(avg, epoch_updates)
            model.load_state_dict(avg["weights"])

        # average and save losses globallyPredict
        all_training_losses[str(self.agent.args.epoch)] = sum(losses.values()) / len(losses)
        self.set("all_training_losses", all_training_losses)

        # save updated global model
        self.set("model", model)

        fsm_logger.info(self.agent.name + ": averaged")
        self.set_next_state(config["server"]["predict"])


class PredictState(State):
    async def run(self):
        # set local variables
        all_labels = []
        all_predictions = []
        all_testing_losses = self.get("all_testing_losses")
        criterion = self.get("criterion")
        device = self.get("device")
        model = self.get("model")
        testing_losses = []
        x_test = self.get("x_test")
        y_test = self.get("y_test")
        y_test_original_labels = self.get("y_test_original_labels")

        print("-    The agent " + self.agent.name + "  predicts at the epoch " + str(self.agent.args.epoch))
        time.sleep(1)

        # predict with model
        Learning.predicting(all_labels, all_predictions, criterion, device, model, testing_losses, x_test,
                   y_test_original_labels, y_test)

        all_testing_losses[str(self.agent.args.epoch)] = sum(testing_losses) / len(testing_losses)

        # save label and predictions as global variables
        self.set("all_labels", all_labels)
        self.set("all_predictions", all_predictions)
        self.set("all_testing_losses", all_testing_losses)

        fsm_logger.info(self.agent.name + ": predicted")
        self.set_next_state(config["server"]["calculate_metrics"])


class CalculateMetricsState(State):
    async def run(self):
        # set local variables
        all_labels = self.get("all_labels")
        all_predictions = self.get("all_predictions")
        all_test_accuracies = self.get("all_test_accuracies")
        all_test_f1_scores = self.get("all_test_f1_scores")
        all_test_precisions = self.get("all_test_precisions")
        all_test_recalls = self.get("all_test_recalls")
        f1_scores_per_classes = self.get("f1_scores_per_classes")
        precisions_per_classes = self.get("precisions_per_classes")
        recalls_per_classes = self.get("recalls_per_classes")

        print("-    The agent " + self.agent.name +
              "  calculates the accuracy, f1-score, precision and recall at the epoch "
              + str(self.agent.args.epoch))

        # calculate metrics
        Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                          all_test_precisions, all_test_recalls, self.agent.args.epoch)
        Metrics.calculate_f1_score_per_classes(all_labels, all_predictions, self.agent.args.epoch, f1_scores_per_classes)
        Metrics.calculate_precisions_per_classes(all_labels, all_predictions, self.agent.args.epoch, precisions_per_classes)
        Metrics.calculate_recalls_per_classes(all_labels, all_predictions, self.agent.args.epoch, recalls_per_classes)

        # save metrics as global variables
        self.set("all_test_accuracies", all_test_accuracies)
        self.set("all_test_f1_scores", all_test_f1_scores)
        self.set("all_test_precisions", all_test_precisions)
        self.set("all_test_recalls", all_test_recalls)
        self.set("f1_scores_per_classes", f1_scores_per_classes)
        self.set("precisions_per_classes", precisions_per_classes)
        self.set("recalls_per_classes", recalls_per_classes)

        '''
        If it's not the last global epoch, proceed to the next one. 
        Otherwise, store the collected metrics. 
        For FL, the next epoch starts by sending global weights.
        However, with ML it starts with training the global model.
        If wait_until_threshold_is_reached has set a threshold,
        then the training ends only if this threshold is reached.
        '''
        fsm_logger.info(self.agent.name + ": metrics calculated")
        if self.agent.args.wait_until_threshold_is_reached == "acc":
            if all_test_accuracies[str(self.agent.args.epoch)] > self.agent.args.threshold:
                self.set_next_state(config["server"]["store_metrics"])
            else:
                self.agent.args.epoch += 1
                if self.agent.args.algorithm == "ML":
                    self.set_next_state(config["server"]["train"])
                else:
                    self.set_next_state(config["server"]["control_agents_present"])
        elif self.agent.args.wait_until_threshold_is_reached == "f1":
            if all_test_f1_scores[str(self.agent.args.epoch)] > self.agent.args.threshold:
                self.set_next_state(config["server"]["store_metrics"])
            else:
                self.agent.args.epoch += 1
                if self.agent.args.algorithm == "ML":
                    self.set_next_state(config["server"]["train"])
                else:
                    self.set_next_state(config["server"]["control_agents_present"])
        elif self.agent.args.wait_until_threshold_is_reached == "pre":
            if all_test_precisions[str(self.agent.args.epoch)] > self.agent.args.threshold:
                self.set_next_state(config["server"]["store_metrics"])
            else:
                self.agent.args.epoch += 1
                if self.agent.args.algorithm == "ML":
                    self.set_next_state(config["server"]["train"])
                else:
                    self.set_next_state(config["server"]["control_agents_present"])
        elif self.agent.args.wait_until_threshold_is_reached == "rec":
            if all_test_recalls[str(self.agent.args.epoch)] > self.agent.args.threshold:
                self.set_next_state(config["server"]["store_metrics"])
            else:
                self.agent.args.epoch += 1
                if self.agent.args.algorithm == "ML":
                    self.set_next_state(config["server"]["train"])
                else:
                    self.set_next_state(config["server"]["control_agents_present"])
        else:
            if self.agent.args.global_epochs > self.agent.args.epoch:
                self.agent.args.epoch += 1
                if self.agent.args.algorithm == "ML":
                    self.set_next_state(config["server"]["train"])
                else:
                    self.set_next_state(config["server"]["control_agents_present"])
            else:
                self.set_next_state(config["server"]["store_metrics"])


class StoreMetricsState(State):
    async def run(self):
        # set local variables
        all_test_accuracies = self.get("all_test_accuracies")
        all_test_f1_scores = self.get("all_test_f1_scores")
        all_testing_losses = self.get("all_testing_losses")
        all_test_precisions = self.get("all_test_precisions")
        all_test_recalls = self.get("all_test_recalls")
        all_training_losses = self.get("all_training_losses")
        f1_scores_per_classes = self.get("f1_scores_per_classes")
        precisions_per_classes = self.get("precisions_per_classes")
        recalls_per_classes = self.get("recalls_per_classes")

        print("-    The agent " + self.agent.name + " stores the metrics at the epoch " + str(self.agent.args.epoch))

        # show last epoch if the threshold is reached
        if self.agent.args.wait_until_threshold_is_reached != "no threshold":
            print("Epochs needed to reach the threshold " + str(self.agent.args.threshold) + ": " +
                  str(self.agent.args.epoch))

        # store metrics
        Metrics.store_metrics(self.agent.name, all_test_accuracies, all_test_f1_scores, all_test_precisions, all_test_recalls,
                      all_testing_losses, all_training_losses, self.agent.args,
                      self.agent.batch_sizes_training_by_non_iid, f1_scores_per_classes, precisions_per_classes,
                      recalls_per_classes)

        fsm_logger.info(self.agent.name + ": metrics stored")
