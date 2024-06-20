from Neuron import Neuron

class Input:
    def __init__(self, size,  layer="Input", layer_after=None):
        self.size = size
        self.layer = layer
        self.dimensions = len(self.size)
        if self.dimensions == 1:
            self.neurons = []
        elif self.dimensions == 2:
            self.neurons = [[]]
        elif self.dimensions == 3:
            self.neurons = [[[]]]

        self.layer_after = layer_after

    def build_layer(self):
        pass


    def build_neurons(self):
        if self.dimensions == 1:
            for i in range(self.size[0]):
                self.neurons.append(Neuron(layer=self.layer))
        elif self.dimensions == 2:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    self.neurons[i].append(Neuron(layer=self.layer))
        elif self.dimensions == 3:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    for k in range(self.size[2]):
                        self.neurons[i][j].append(Neuron(layer=self.layer))
