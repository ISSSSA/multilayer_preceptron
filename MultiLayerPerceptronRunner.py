from core.MultiLayerPerceptron import MultiLayerPerceptron
from domain.ClassLabelMapping import ClassLabelMapping
from util.MatrixCsvLoader import MatrixCsvLoader
from loss.CrossEntropyLossFunction import CrossEntropyLossFunction  # Импорт функции потерь


class MultiLayerPerceptronRunner:
    def __init__(self):
        self.input_size = 32 * 32
        self.hidden_layer_sizes = [128, 64]
        self.output_size = 10
        self.learning_rate = 0.2

        # Передаем функцию потерь при создании экземпляра
        self.loss_function = CrossEntropyLossFunction()  # Создание экземпляра функции потерь
        self.mlp = MultiLayerPerceptron(
            self.input_size,
            self.hidden_layer_sizes,
            self.output_size,
            self.learning_rate,
            self.loss_function  # Добавлен аргумент функции потерь
        )

    def run(self):
        training_data = self.load_data(r"C:\\Users\\Воронов Игорь\\Documents\\Dataset\\output_hyped.csv")
        self.mlp.train(training_data)

    def load_data(self, file_path):
        matrix_csv_loader = MatrixCsvLoader(file_path)
        return matrix_csv_loader.load_data()

    def test_model(self, validation_data):
        for input_data in validation_data:
            predict = self.mlp.predict(input_data.data)
            true_label = ClassLabelMapping.from_input_data(input_data)  # Правильное использование метода
            print(f'True label: {true_label}, Predicted: {predict}')

# Инициализация и запуск
