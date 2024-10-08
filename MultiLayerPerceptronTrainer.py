import random
from core.MultiLayerPerceptron import MultiLayerPerceptron
from domain.ClassLabelMapping import ClassLabelMapping
from util.MatrixCsvLoader import MatrixCsvLoader
from loss.CrossEntropyLossFunction import CrossEntropyLossFunction

class MultiLayerPerceptronTrainer:
    def __init__(self):
        self.input_size = 32 * 32
        self.hidden_layer_sizes = [128, 64]
        self.output_size = 10
        self.learning_rate = 0.2

        self.loss_function = CrossEntropyLossFunction()
        self.mlp = MultiLayerPerceptron(
            self.input_size,
            self.hidden_layer_sizes,
            self.output_size,
            self.learning_rate,
            self.loss_function
        )

    def run_training(self, training_data_path):
        training_data = self.load_training_inputs(training_data_path)
        self.mlp.train(training_data)

    def load_training_inputs(self, file_path):
        matrix_csv_loader = MatrixCsvLoader(file_path)
        return matrix_csv_loader.load_data()
