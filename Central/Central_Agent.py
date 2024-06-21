import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import *
import spade
from torch import nn
import config
from models import CNN
import dataset


class Central_Agent(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state(State):
    async def run(self):
        print("-    This is the setup state")
        global all_accuracies, all_f1_scores, all_gradients, all_labels, all_precisions, all_predictions, all_recalls, all_weights, batch_size, criterion, device, epoch, global_epochs, learning_rate, local_epochs, model, optimizer, X_train, Y_train, X_test, Y_test
        all_accuracies = []
        all_f1_scores = []
        all_gradients = []
        all_labels = []
        all_precisions = []
        all_predictions = []
        all_recalls = []
        all_weights = []
        batch_size = 3000
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epoch = 0
        global_epochs = 10
        learning_rate = 0.1
        local_epochs = 5
        model = CNN(10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        dataset.walk_to_the_right_directory()
        X_train, Y_train = dataset.prepare_train(batch_size, global_epochs, local_epochs, (batch_size, 28, 28))
        X_test, Y_test = dataset.prepare_test(batch_size, global_epochs, local_epochs, (batch_size, 28, 28))
        model.to(device)
        self.set_next_state(config.TRAIN_STATE)


class train_state(State):
    async def run(self):
        print("-    This is the train state")
        global all_gradients, device, epoch, model, X_train, Y_train
        model.train()

        for images, labels in zip(X_train[epoch], Y_train[epoch]):
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels)
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            all_gradients.append(optimizer.state_dict())
            optimizer.step()

        all_weights.append(model.state_dict())
        self.set_next_state(config.PREDICT_STATE)


class predict_state(State):
    async def run(self):
        print("-    This is the predict state")
        global all_labels, all_predictions, criterion, device, epoch, model, X_train, Y_train
        model.eval()

        with torch.no_grad():
            for images, labels in zip(X_train[epoch], Y_train[epoch]):
                images = torch.from_numpy(images)
                labels = torch.from_numpy(labels)
                labels = labels.type(torch.LongTensor)
                outputs = model(images)
                losses = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        self.set_next_state(config.COLLECT_METRICS_STATE)


class collect_metrics_state(State):
    async def run(self):
        print("-    This is the collect metrics state")
        global all_accuracies, all_f1_scores, all_labels, all_precisions, all_predictions, all_recalls, epoch, global_epochs, model

        all_accuracies.append(accuracy_score(all_labels, all_predictions))
        all_precisions.append(
            precision_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))
        all_recalls.append(
            recall_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))
        all_f1_scores.append(
            f1_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))

        epoch += 1

        if global_epochs > epoch:
            self.set_next_state(config.TRAIN_STATE)
        else:
            self.set_next_state(config.EVALUATE_STATE)


class evaluate_state(State):
    async def run(self):
        print("-    This is the evaluate state")
        global all_accuracies, all_f1_scores, all_precisions, all_recalls
        df = pd.DataFrame(
            {'accuracy': all_accuracies, 'precision': all_precisions, 'recall': all_recalls, 'f1': all_f1_scores},
            index=np.arange(len(all_accuracies)))
        df.plot(title="Metrics Plot")
        plt.show()


class Server(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        fsm = Central_Agent()

        fsm.add_state(name=config.SETUP_STATE, state=setup_state(), initial=True)
        fsm.add_state(name=config.TRAIN_STATE, state=train_state())
        fsm.add_state(name=config.PREDICT_STATE, state=predict_state())
        fsm.add_state(name=config.COLLECT_METRICS_STATE, state=collect_metrics_state())
        fsm.add_state(name=config.EVALUATE_STATE, state=evaluate_state())

        fsm.add_transition(config.SETUP_STATE, config.TRAIN_STATE)
        fsm.add_transition(config.TRAIN_STATE, config.PREDICT_STATE)
        fsm.add_transition(config.PREDICT_STATE, config.COLLECT_METRICS_STATE)
        fsm.add_transition(config.COLLECT_METRICS_STATE, config.TRAIN_STATE)
        fsm.add_transition(config.COLLECT_METRICS_STATE, config.EVALUATE_STATE)

        self.add_behaviour(self.ServerBehaviour())
        self.add_behaviour(fsm)

    class ServerBehaviour(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's approve it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)
            self.presence.subscribe(jid)

        async def run(self):
            self.presence.set_available()
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available


async def main():
    jid_server = "server@localhost"
    passwd_server = "server"
    server = Server(jid_server, passwd_server)
    await server.start()

    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break
    await server.stop()


if __name__ == "__main__":
    spade.run(main())
