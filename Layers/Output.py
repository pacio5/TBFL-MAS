from Connection import Connection
from Neuron import Neuron
from numba import njit, prange


class Output:
    def __init__(self, size, activation_function, learning_rate, layer="Output", layer_before=None):
        self.size = size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.layer = layer
        self.dimensions = len(self.size)
        if self.dimensions == 1:
            self.neurons = []
        elif self.dimensions == 2:
            self.neurons = [[]]
        elif 3 == self.dimensions:
            self.neurons = [[[]]]
        self.layer_before = layer_before
        self.input_connections = []

    def build_layer(self):
        if self.dimensions == 1:
            for neuron_layer in self.neurons:
                if self.layer_before.dimensions == 1:
                    for neuron_layer_before in self.layer_before.neurons:
                        connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                        neuron_layer_before.output_connections.append(connection)
                        neuron_layer.input_connections.append(connection)
                        self.input_connections.append(connection)
                elif self.layer_before.dimensions == 2:
                    for neuron_layer_before_list in self.layer_before.neurons:
                        for neuron_layer_before in neuron_layer_before_list:
                            connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                            neuron_layer_before.output_connections.append(connection)
                            neuron_layer.input_connections.append(connection)
                            self.input_connections.append(connection)
                    for neuron_layer_before_list1 in self.layer_before.neurons:
                        for neuron_layer_before_list2 in neuron_layer_before_list1:
                            for neuron_layer_before in neuron_layer_before_list2:
                                connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                                neuron_layer_before.output_connections.append(connection)
                                neuron_layer.input_connections.append(connection)
                                self.input_connections.append(connection)

        elif self.dimensions == 2:
            for neuron_layer_list in self.neurons:
                for neuron_layer in neuron_layer_list:
                    if self.layer_before.dimensions == 1:
                        for neuron_layer_before in self.layer_before.neurons:
                            connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                            neuron_layer_before.output_connections.append(connection)
                            neuron_layer.input_connections.append(connection)
                            self.input_connections.append(connection)
                    elif self.layer_before.dimensions == 2:
                        for neuron_layer_before_list in self.layer_before.neurons:
                            for neuron_layer_before in neuron_layer_before_list:
                                connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                                neuron_layer_before.output_connections.append(connection)
                                neuron_layer.input_connections.append(connection)
                                self.input_connections.append(connection)
                    elif self.layer_before.dimensions == 3:
                        for neuron_layer_before_list1 in self.layer_before.neurons:
                            for neuron_layer_before_list2 in neuron_layer_before_list1:
                                for neuron_layer_before in neuron_layer_before_list2:
                                    connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                                    neuron_layer_before.output_connections.append(connection)
                                    neuron_layer.input_connections.append(connection)
                                    self.input_connections.append(connection)

        elif self.dimensions == 3:
            for neuron_layer_list1 in self.neurons:
                for neuron_layer_list2 in neuron_layer_list1:
                    for neuron_layer in neuron_layer_list2:
                        if self.layer_before.dimensions == 1:
                            for neuron_layer_before in self.layer_before.neurons:
                                connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                                neuron_layer_before.output_connections.append(connection)
                                neuron_layer.input_connections.append(connection)
                                self.input_connections.append(connection)
                        elif self.layer_before.dimensions == 2:
                            for neuron_layer_before_list in self.layer_before.neurons:
                                for neuron_layer_before in neuron_layer_before_list:
                                    connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                                    neuron_layer_before.output_connections.append(connection)
                                    neuron_layer.input_connections.append(connection)
                                    self.input_connections.append(connection)
                        elif self.layer_before.dimensions == 3:
                            for neuron_layer_before_list1 in self.layer_before.neurons:
                                for neuron_layer_before_list2 in neuron_layer_before_list1:
                                    for neuron_layer_before in neuron_layer_before_list2:
                                        connection = Connection(neuron_layer_before, neuron_layer, self.learning_rate)
                                        neuron_layer_before.output_connections.append(connection)
                                        neuron_layer.input_connections.append(connection)
                                        self.input_connections.append(connection)

    def build_neurons(self):
        if self.dimensions == 1:
            for i in range(self.size[0]):
                self.neurons.append(Neuron(self.activation_function, layer=self.layer))
        elif self.dimensions == 2:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    self.neurons[i].append(Neuron(self.activation_function, layer=self.layer))
        elif self.dimensions == 3:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    for k in range(self.size[2]):
                        self.neurons[i][j].append(Neuron(self.activation_function, layer=self.layer))
