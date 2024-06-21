from Layers.Input import Input
from Layers.Output import Output
from Layers.Dense import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, size_input, hidden_layers, size_output, activation_function_output, learning_rate_output):
        np.random.seed(0)
        self.layers = []
        self.layers.append(Input(size_input))
        for hidden_layer in hidden_layers:
            self.layers.append(hidden_layer)
        self.layers.append(Output(size_output, activation_function_output, learning_rate_output))
        self.predictions = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def build_model(self):
        self.layers[0].layer_after = self.layers[1]
        self.layers[-1].layer_before = self.layers[-2]

        for i in range(1, len(self.layers) - 1):
            self.layers[i].layer_after = self.layers[i + 1]
            self.layers[i].layer_before = self.layers[i - 1]

        for layer in self.layers:
            layer.build_neurons()

        for i in range(1, len(self.layers)):
            self.layers[i].build_layer()

    def train(self, train_inputs, train_outputs):

        for i in range(len(train_inputs)):
            if self.layers[0].dimensions == 1:
                for j in range(len(self.layers[0].neurons)):
                    self.layers[0].neurons[j].output = train_inputs[i][j]
            elif self.layers[0].dimensions == 2:
                for j in range(len(self.layers[0].neurons)):
                    for k in range(len(self.layers[0].neurons[j])):
                        self.layers[0].neurons[j][k].output = train_inputs[i][j][k]
            elif self.layers[0].dimensions == 3:
                for j in range(len(self.layers[0].neurons)):
                    for k in range(len(self.layers[0].neurons[j])):
                        for l in range(len(self.layers[0].neurons[j][k])):
                            self.layers[0].neurons[j][k][l].output = train_inputs[i][j][k][l]

            for j in range(1, len(self.layers)):
                if self.layers[j].dimensions == 1:
                    for k in range(len(self.layers[j].neurons)):
                        self.layers[j].neurons[k].forward_propagation()
                elif self.layers[j].dimensions == 2:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            self.layers[j].neurons[k][l].forward_propagation()
                elif self.layers[j].dimensions == 3:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            for m in range(len(self.layers[j].neurons[k][l])):
                                self.layers[j].neurons[k][l][m].forward_propagation()

            for j in range(len(self.layers) - 1, 1, -1):
                if self.layers[j].dimensions == 1:
                    for k in range(len(self.layers[j].neurons)):
                        self.layers[j].neurons[k].back_propagation_error(train_outputs[i])
                elif self.layers[j].dimensions == 2:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            self.layers[j].neurons[k][l].back_propagation_error(train_outputs[i])
                elif self.layers[j].dimensions == 3:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            for m in range(len(self.layers[j].neurons[k][l])):
                                self.layers[j].neurons[k][l][m].back_propagation_error(train_outputs[i])

            for j in range(1, len(self.layers)):
                for connection in self.layers[j].input_connections:
                    connection.gradient_update()
                    connection.weight_update()

            for j in range(1, len(self.layers)):
                if self.layers[j].dimensions == 1:
                    for k in range(len(self.layers[j].neurons)):
                        self.layers[j].neurons[k].update_bias()
                elif self.layers[j].dimensions == 2:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            self.layers[j].neurons[k][l].update_bias()
                elif self.layers[j].dimensions == 3:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            for m in range(len(self.layers[j].neurons[k][l])):
                                self.layers[j].neurons[k][l][m].update_bias()

    def predict(self, test_inputs):
        self.predictions = []
        for i in range(len(test_inputs)):
            if self.layers[0].dimensions == 1:
                for j in range(len(self.layers[0].neurons)):
                    self.layers[0].neurons[j].output = test_inputs[i][j]
            elif self.layers[0].dimensions == 2:
                for j in range(len(self.layers[0].neurons)):
                    for k in range(len(self.layers[0].neurons[j])):
                        self.layers[0].neurons[j][k].output = test_inputs[i][j][k]
            elif self.layers[0].dimensions == 3:
                for j in range(len(self.layers[0].neurons)):
                    for k in range(len(self.layers[0].neurons[j])):
                        for l in range(len(self.layers[0].neurons[j][k])):
                            self.layers[0].neurons[j][k][l].output = test_inputs[i][j][k][l]

            for j in range(1, len(self.layers)):
                if 1 == self.layers[j].dimensions:
                    for k in range(len(self.layers[j].neurons)):
                        self.layers[j].neurons[k].forward_propagation()
                        if j == len(self.layers) - 1:
                            self.predictions.append(self.layers[j].neurons[k].output)
                elif self.layers[j].dimensions == 2:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            self.layers[j].neurons[k][l].forward_propagation()
                elif self.layers[j].dimensions == 3:
                    for k in range(len(self.layers[j].neurons)):
                        for l in range(len(self.layers[j].neurons[k])):
                            for m in range(len(self.layers[j].neurons[k][l])):
                                self.layers[j].neurons[k][l][m].forward_propagation()

    def collect_metrics(self, test_outputs):
        self.accuracies.append(accuracy_score(test_outputs, self.predictions))
        self.precisions.append(
            precision_score(test_outputs, self.predictions, average="weighted", labels=np.unique(self.predictions)))
        self.recalls.append(
            recall_score(test_outputs, self.predictions, average="weighted", labels=np.unique(self.predictions)))
        self.f1_scores.append(
            f1_score(test_outputs, self.predictions, average="weighted", labels=np.unique(self.predictions)))

    def evaluate(self):
        df = pd.DataFrame(
            {'accuracy': self.accuracies, 'precision': self.precisions, 'recall': self.recalls, 'f1': self.f1_scores},
            index=np.arange(len(self.accuracies)))
        df.plot(title="Metrics Plot")
        plt.show()
