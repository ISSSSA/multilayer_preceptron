from activasion.SigmoidActivationFunctionFunction import SigmoidActivationFunction
from loss.CrossEntropyLossFunction import CrossEntropyLossFunction
from activasion.ActivationFunction import ActivationFunction
from loss.LossFunction import LossFunction


class MultiLayerPerceptronConfiguration:
    default_activation_function: ActivationFunction = SigmoidActivationFunction()
    default_loss_function: LossFunction = CrossEntropyLossFunction()

    EPOCH_COUNT: int = 10
