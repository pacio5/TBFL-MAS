import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import *
import spade
from torch import nn
from models import CNN
import dataset
import yaml

with open("../config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)


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
        self.set_next_state(config["server"]["train"])


class train_state(State):
    async def run(self):
        global all_gradients, all_weights, epoch, model, optimizer, X_train, Y_train
        print("-    This is the train state")
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

        self.set_next_state(config["server"]["predict"])


class send_state(State):
    async def run(self):
        global model

        self.set_next_state(config["server"]["receive"])


class receive_state(State):
    async def run(self):
        global model

        self.set_next_state(config["server"]["avg"])


class avg_state(State):
    async def run(self):
        global model

        self.set_next_state(config["server"]["predict"])

class predict_state(State):
    async def run(self):
        global all_labels, all_predictions, epoch, model, X_test, Y_test
        print("-    This is the predict state")
        model.eval()

        with torch.no_grad():
            for images, labels in zip(X_test[epoch], Y_test[epoch]):
                images = torch.from_numpy(images)
                labels = torch.from_numpy(labels)
                labels = labels.type(torch.LongTensor)
                outputs = model(images)
                losses = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        self.set_next_state(config["server"]["collect_metrics"])


class collect_metrics_state(State):
    async def run(self):
        global all_accuracies, all_f1_scores, all_labels, all_precisions, all_predictions, all_recalls, epoch, global_epochs
        print("-    This is the collect metrics state")

        all_accuracies.append(accuracy_score(all_labels, all_predictions))
        all_precisions.append(
            precision_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))
        all_recalls.append(
            recall_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))
        all_f1_scores.append(
            f1_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions)))

        epoch += 1

        if global_epochs > epoch:
            self.set_next_state(config["server"]["train"])
        else:
            self.set_next_state(config["server"]["plot_metrics"])


class plot_metrics_state(State):
    async def run(self):
        global all_accuracies, all_f1_scores, all_precisions, all_recalls
        print("-    This is the evaluate state")
        df = pd.DataFrame(
            {'accuracy': all_accuracies, 'precision': all_precisions, 'recall': all_recalls, 'f1': all_f1_scores},
            index=np.arange(len(all_accuracies)))
        df.plot(title="Metrics Plot")
        plt.show()



class Server(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        fsm = Central_Agent()

        fsm.add_state(name=config["server"]["set_up"], state=setup_state(), initial=True)
        fsm.add_state(name=config["server"]["train"], state=train_state())
        fsm.add_state(name=config["server"]["send"], state=send_state())
        fsm.add_state(name=config["server"]["receive"], state=receive_state())
        fsm.add_state(name=config["server"]["avg"], state=avg_state())
        fsm.add_state(name=config["server"]["predict"], state=predict_state())
        fsm.add_state(name=config["server"]["collect_metrics"], state=collect_metrics_state())
        fsm.add_state(name=config["server"]["plot_metrics"], state=plot_metrics_state())

        fsm.add_transition(config["server"]["set_up"], config["server"]["train"])
        fsm.add_transition(config["server"]["set_up"], config["server"]["send"])
        fsm.add_transition(config["server"]["train"], config["server"]["predict"])
        fsm.add_transition(config["server"]["send"], config["server"]["receive"])
        fsm.add_transition(config["server"]["receive"], config["server"]["avg"])
        fsm.add_transition(config["server"]["avg"], config["server"]["predict"])
        fsm.add_transition(config["server"]["predict"], config["server"]["collect_metrics"])
        fsm.add_transition(config["server"]["collect_metrics"], config["server"]["train"])
        fsm.add_transition(config["server"]["collect_metrics"], config["server"]["plot_metrics"])

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
