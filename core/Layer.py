import numpy as np
from core.Neuron import Neuron


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = np.array([Neuron(input_size) for _ in range(output_size)])
        self.outputs = np.zeros(output_size)

    def forward(self, inputs):
        self.outputs = np.array([neuron.activate(inputs) for neuron in self.neurons])
        return self.outputs

    def compute_output_deltas(self, target_label):
        for i, neuron in enumerate(self.neurons):
            target = 1.0 if i == target_label else 0.0
            neuron.compute_output_delta(target)

    def compute_hidden_deltas(self, next_layer):
        for i, neuron in enumerate(self.neurons):
            neuron.compute_hidden_delta(next_layer.get_neurons(), i)

    def update_weights(self, inputs, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(inputs, learning_rate)

    def get_neurons(self):
        return self.neurons

    def get_outputs(self):
        return self.outputs
