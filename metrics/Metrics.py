class Metrics:
    def __init__(self, accuracy, precision, recall, loss_function_value):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.loss_function_value = loss_function_value

    def format_metrics(self):
        return (
            "Metrics{accuracy=%.2f, precision=%.2f, recall=%.2f, lossFunctionValue=%.2f}"
            % (self.accuracy, self.precision, self.recall, self.loss_function_value)
        )
