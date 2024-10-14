from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def calculate_loss(self, predicted, target_label):
        pass
