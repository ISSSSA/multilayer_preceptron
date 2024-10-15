import random
from typing import List
from core.Layer import Layer

from configuration.MultiLayerPerceptronConfiguration import MultiLayerPerceptronConfiguration

from domain.ClassLabelMapping import ClassLabelMapping
from domain.InputData import InputData
from loss.LossFunction import LossFunction
from metrics.MetricsBuilder import MetricsBuilder
import time


class MultiLayerPerceptron:
    def __init__(self, input_size: int, hidden_layer_sizes: List[int], output_size: int,
                 learning_rate: float, loss_function: LossFunction, config: MultiLayerPerceptronConfiguration):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.config = config
        previous_size = input_size
        for size in hidden_layer_sizes:
            self.layers.append(Layer(previous_size, size))
            previous_size = size
        self.layers.append(Layer(previous_size, output_size))
        print(self.layers)

    def forward(self, inputs: List[float]) -> List[float]:
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backpropagation(self, target_label: int, inputs: List[float]):
        self.layers[-1].compute_output_deltas(target_label)

        for l in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[l]
            next_layer = self.layers[l + 1]
            current_layer.compute_hidden_deltas(next_layer)

        layer_inputs = inputs
        for layer in self.layers:
            layer.update_weights(layer_inputs, self.learning_rate)
            layer_inputs = layer.get_outputs()

    def train(self, dataset: List[InputData]):
        metrics_builder = MetricsBuilder()
        for epoch in range(1, self.config.EPOCH_COUNT + 1):
            total_loss = 0.0
            random.shuffle(dataset)
            print(f"Epoch {epoch}:")
            for input_data in dataset:
                inputs = input_data.data
                self.forward(inputs)
                self.backpropagation(ClassLabelMapping.from_input_data(input_data), inputs)
                guess = self.max_idx(self.layers[-1].get_outputs())
                metrics_builder.add_value(self.layers[-1].get_outputs(), guess)

                total_loss += self.compute_loss(ClassLabelMapping.from_input_data(input_data))

            metrics = metrics_builder.add_loss_function_value(total_loss / len(dataset)).create_metrics()
            print(metrics.format_metrics())
            metrics_builder.clear()

    def compute_loss(self, target_label: int) -> float:
        predicted = self.layers[-1].get_outputs()
        return self.loss_function.calculate_loss(predicted, target_label)

    def predict(self, inputs):
        return self.max_idx(self.forward(inputs))

    @staticmethod
    def max_idx(arr) -> int:
        return list(arr).index(max(arr))
