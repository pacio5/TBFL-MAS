import numpy as np

class Neuron:
    def __init__(self, activation_function=None, layer=None):
        self.input_connections = []
        self.output_connections = []
        self.output = 0
        self.bias = np.random.rand()
        self.activation_function = activation_function
        self.error = 0
        self.layer = layer

    def forward_propagation(self):
        if self.layer != "Input":
            for i in range(len(self.input_connections)):
                self.output += self.input_connections[i].weight * self.input_connections[i].begin.output
            self.output += self.bias
            if self.activation_function == 'sigmoid':
                self.output = 1 / (1 + np.exp(-self.output))
            elif self.activation_function == 'tanh':
                self.output = 2 / (1 + np.exp(-2 * self.output)) - 1
            elif self.activation_function == 'relu':
                self.output = np.maximum(0, self.output)

    def back_propagation_error(self, x):
        if self.layer != "Input":
            self.error = 0
            if len(self.output_connections) != 0:
                for i in range(len(self.output_connections)):
                    self.error += self.output_connections[i].weight * self.output_connections[i].end.error
            else:
                self.error = x - self.output

    def update_bias(self):
        if self.layer != "Input":
            if self.error > 0 and self.bias < 100:
                self.bias += 1
            elif self.bias > -100:
                self.bias -= 1
            #print(self.bias)
