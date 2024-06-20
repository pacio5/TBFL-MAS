import numpy as np


class Connection:
    def __init__(self, begin, end, learning_rate):
        self.begin = begin
        self.end = end
        self.weight = np.random.rand()
        self.learning_rate = learning_rate
        self.gradient = 0

    def gradient_update(self):
        if self.end.activation_function == 'sigmoid':
            self.gradient = (1 / (1 + np.exp(-self.end.output)) * (
                        1 - 1 / (1 + np.exp(-self.end.output)))
                        * self.learning_rate * self.end.error * self.begin.output)
        elif self.end.activation_function == 'tanh':
            self.gradient = (1 - np.power(np.exp(self.end.output) - np.exp(-self.end.output), 2) / np.power(
                np.exp(self.end.output) + np.exp(-self.end.output), 2)
                * self.learning_rate * self.end.error * self.begin.output)
        elif self.end.activation_function == 'relu':
            self.gradient = np.maximum(0, self.learning_rate * self.end.error * self.begin.output)
        else:
            self.gradient = self.learning_rate * self.end.error * self.begin.output

    def weight_update(self):
        if self.end.error > 0 and self.weight < 1000:
            self.weight += self.gradient
        elif self.weight > -1000:
            self.weight -= self.gradient
        """print("gradient: " + str(self.gradient))
        print("weight: " + str(self.weight))"""

