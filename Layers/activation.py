import numpy as np


class Activation:
    def __init__(self, activation_function):
        self.param = {}
        self.param['activation_function'] = activation_function
    
    def forward_propagation(self, inputs):
        self.param['inputs'] = inputs
        activation_function = self.param['activation_function']
        if activation_function == 'sigmoid':
            outputs = 1 / (1 + np.exp(-inputs))
        elif activation_function == 'tanh':
            outputs = 2 / (1 + np.exp(-2 * inputs)) - 1
        elif activation_function == 'relu':
            outputs = np.where(inputs >= 0, inputs, 0)
    
        return outputs
    
    def backward_propagation(self, errors):
        inputs = self.param['inputs']
        activation_function = self.param['activation_function']
        if activation_function == 'sigmoid':
            outputs = errors * (1 / (1 + np.exp(-inputs)) * (1 - 1 / (1 + np.exp(-inputs))))
        elif activation_function == 'tanh':
            outputs = errors * (1 - np.power(np.exp(inputs) - np.exp(-inputs), 2) / np.power(np.exp(inputs) + np.exp(-inputs), 2))
        elif activation_function == 'relu':
            outputs = errors * np.where(inputs >= 0, 1, 0)
    
        return outputs
        