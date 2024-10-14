import numpy as np


class CrossEntropyLossFunction:
    EPSILON = 1e-15

    def calculate_loss(self, predicted: list, target_label: int):
        total_loss = 0.0
        for i in range(len(predicted)):
            target_value = 1.0 if i == target_label else 0.0
            log = np.log(predicted[i] + self.EPSILON)
            total_loss -= target_value * log
        return total_loss
