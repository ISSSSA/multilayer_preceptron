import random

from activasion.ActivationFunction import ActivationFunction
from configuration.MultiLayerPerceptronConfiguration import MultiLayerPerceptronConfiguration


class Neuron:
    def __init__(self, input_size: int, activation_function: ActivationFunction = None):
        self.bias = None
        self.weights = None
        self.input_size = input_size
        self.activation_function = activation_function or MultiLayerPerceptronConfiguration.default_activation_function
        self.initialize_weights_and_bias()
        self.output = 0.0
        self.input_sum = 0.0
        self.delta = 0.0

    def initialize_weights_and_bias(self):
        self.weights = [random.gauss(0, 0.1) for _ in
                        range(self.input_size)]
        self.bias = 0.0

    def activate(self, inputs: list) -> float:
        z = self.bias + sum(w * i for w, i in zip(self.weights, inputs))
        self.output = self.activation_function.apply(z)
        return self.output

    def compute_output_delta(self, target: float):
        self.delta = self.output - target

    def compute_hidden_delta(self, next_layer_neurons, index: int):
        sum_delta = sum(neuron.weights[index] * neuron.delta for neuron in next_layer_neurons)
        self.delta = sum_delta * self.activation_function.apply_derivative(self.output)

    def update_weights(self, inputs: list, learning_rate: float):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.delta * inputs[i]
        self.bias -= learning_rate * self.delta
