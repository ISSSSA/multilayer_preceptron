import numpy as np


class ActivationFunction:
    def apply(self, value: float) -> float:
        raise NotImplementedError("Must implement apply method")

    def apply_derivative(self, value: float) -> float:
        raise NotImplementedError("Must implement apply_derivative method")

    def cross_entropy_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        rows, cols = predictions.shape
        loss = 0.0

        # Adding a small constant (1e-15) to avoid log(0)
        for i in range(rows):
            for j in range(cols):
                if labels[i][j] == 1.0:
                    loss -= np.log(predictions[i][j] + 1e-15)

        return loss / rows
