from Layers.Input import Input
from Layers.Output import Output
from Layers.Dense import Dense
from Layers.activation import Activation
from Layers.fully_connected import FullyConnected
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self):
        self.layers = []
        self.layers.append(FullyConnected(100))
        self.layers.append(Activation("relu"))
        self.layers.append(FullyConnected(1))
        self.layers.append(Activation("sigmoid"))
        self.predictions = []

    def train(self, train_inputs, train_outputs):
        for i in range(len(train_inputs)):
            prediction = train_inputs[i].copy()
            output = train_outputs[i].copy()
            for layer in self.layers:
                prediction = layer.forward_propagation(prediction)
            loss = output - prediction
            for i in range(len(self.layers)-1, 0, -1):
                loss = self.layers[i].backward_propagation(loss)

            for layer in self.layers:
                if 'weights' in layer.param:
                    layer.update(0.1)

    def predict(self, test_inputs):
        self.predictions = []
        for input in test_inputs:
            prediction = input.copy()
            for layer in self.layers:
                prediction = layer.forward_propagation(prediction)
            self.predictions.append(prediction)
        return self.predictions

    def collect_metrics(self, test_outputs):
        self.accuracies.append(accuracy_score(test_outputs, self.predictions))
        self.precisions.append(
            precision_score(test_outputs, self.predictions, average="weighted", labels=np.unique(self.predictions)))
        self.recalls.append(
            recall_score(test_outputs, self.predictions, average="weighted", labels=np.unique(self.predictions)))
        self.f1s.append(
            f1_score(test_outputs, self.predictions, average="weighted", labels=np.unique(self.predictions)))

    def evaluate(self):
        df = pd.DataFrame(
            {'accuracy': self.accuracies, 'precision': self.precisions, 'recall': self.recalls, 'f1': self.f1s},
            index=np.arange(len(self.accuracies)))
        df.plot(title="Metrics Plot")
        plt.show()
