from metrics.MetricFunctions import MetricFunctions
from metrics.Metrics import Metrics

class MetricsBuilder:
    def __init__(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.loss_function_value = 0.0

    def add_value(self, predicted, target_label):
        for i in range(len(predicted)):
            predicted_value = predicted[i]
            if predicted_value > 0.5:
                if i == target_label:
                    self.true_positive += 1
                else:
                    self.false_positive += 1
            else:
                if i != target_label:
                    self.true_negative += 1
                else:
                    self.false_negative += 1
        return self

    def add_loss_function_value(self, value):
        self.loss_function_value = value
        return self

    def clear(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.loss_function_value = 0.0
        return self

    def create_metrics(self):
        recall = MetricFunctions.recall(self.true_positive, self.false_negative)
        accuracy = MetricFunctions.accuracy(
            self.true_positive, self.false_positive, self.false_negative, self.true_negative
        )
        precision = MetricFunctions.precision(self.true_positive, self.false_positive)

        return Metrics(accuracy, precision, recall, self.loss_function_value)
