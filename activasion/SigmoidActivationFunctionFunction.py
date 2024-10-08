import numpy as np

from activasion.ActivationFunction import ActivationFunction


class SigmoidActivationFunction(ActivationFunction):
    def apply(self, value: float) -> float:
        return 1.0 / (1.0 + np.exp(-value))

    def apply_derivative(self, value: float) -> float:
        return value * (1 - value)
