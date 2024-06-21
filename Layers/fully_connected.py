import numpy as np


class FullyConnected:
    def __init__(self, size):
        self.param = {}
        self.size = size

    def forward_propagation(self, inputs):
        self.param["inputs"] = inputs
        self.param["weights"] = np.random.random_sample(size=(inputs.shape[0], self.size))
        self.param["biases"] = np.random.random_sample(size=(self.size, 1))
        return np.dot(inputs, self.param["weights"]) + self.param["biases"]

    def backward_propagation(self, errors):
        self.param["errors"] = errors
        batch_size = errors.shape[1]
        self.param["gradient_weights"] = np.dot(np.transpose(self.param["inputs"]), errors) / batch_size
        self.param["gradient_biases"] = np.sum(errors, axis=1, keepdims=True)

        return np.dot(errors, self.param["weights"].T)

    def update(self, learning_rate=0.1):
        self.param["weights"] -= learning_rate * self.param["gradient_weights"] * self.param["weights"]
        self.param["biases"] -= learning_rate * self.param["gradient_biases"] * self.param["biases"]
