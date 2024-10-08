from core.MultiLayerPerceptron import MultiLayerPerceptron
from domain.ClassLabelMapping import ClassLabelMapping
from util.MatrixCsvLoader import MatrixCsvLoader

class MultiLayerPerceptronTester:
    def __init__(self, mlp: MultiLayerPerceptron):
        self.mlp = mlp

    def run_testing(self, validation_data_path):
        validation_data = self.load_validation_inputs(validation_data_path)
        self.validate(validation_data)

    def load_validation_inputs(self, file_path):
        matrix_csv_loader = MatrixCsvLoader(file_path)
        return matrix_csv_loader.load_data()

    def validate(self, validation_data):
        for input_data in validation_data:
            predict = self.mlp.predict(input_data.data)
            true_label = ClassLabelMapping.from_input_data(input_data)
            print(f'True label: {true_label}, Predicted: {predict}')
