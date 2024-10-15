from activasion.SigmoidActivationFunctionFunction import SigmoidActivationFunction
from loss.CrossEntropyLossFunction import CrossEntropyLossFunction
from activasion.ActivationFunction import ActivationFunction
from loss.LossFunction import LossFunction


class MultiLayerPerceptronConfiguration:
    EPOCH_COUNT = 10
    default_activation_function: ActivationFunction = SigmoidActivationFunction()
    default_loss_function: LossFunction = CrossEntropyLossFunction()

    def __init__(self, epochs: int = 10, ):
        self.EPOCH_COUNT = epochs
