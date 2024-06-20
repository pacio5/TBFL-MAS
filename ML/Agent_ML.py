from spade.agent import Agent
from spade.behaviour import *
import spade
import config
from Model import Model
from Layers.Dense import Dense
import dataset

number_of_epochs = 3
epoch = 0
model = None
hidden_layers = []
dataset.walk_to_the_right_directory()
X_train, Y_train = dataset.prepare_train(3000, number_of_epochs)
X_test, Y_test = dataset.prepare_test(3000, number_of_epochs)
learning_rate=0.1


class Agent_ML(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state(State):
    async def run(self):
        print("-    This is the setup state")
        global model
        hidden_layers.append(Dense(size=[200], learning_rate=learning_rate))
        hidden_layers.append(Dense(size=[200], learning_rate=learning_rate))
        model = Model([764], hidden_layers, [1], "sigmoid", learning_rate_output=learning_rate)
        model.build_model()

        self.set_next_state(config.TRAIN_STATE)


class train_state(State):
    async def run(self):
        print("-    This is the train state")
        global model
        global epoch
        model.train(X_train[epoch], Y_train[epoch])

        self.set_next_state(config.PREDICT_STATE)


class predict_state(State):
    async def run(self):
        print("-    This is the predict state")
        global model
        global epoch
        model.predict(X_test[epoch])

        self.set_next_state(config.COLLECT_METRICS_STATE)


class collect_metrics_state(State):
    async def run(self):
        print("-    This is the collect metrics state")
        global model
        global epoch
        global number_of_epochs
        model.collect_metrics(Y_test[epoch])

        epoch += 1

        if number_of_epochs > epoch:
            self.set_next_state(config.TRAIN_STATE)
        else:
            self.set_next_state(config.EVALUATE_STATE)


class evaluate_state(State):
    async def run(self):
        print("-    This is the evaluate state")
        global model
        model.evaluate()


class Server(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        fsm = Agent_ML()

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

        self.add_behaviour(fsm)


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
