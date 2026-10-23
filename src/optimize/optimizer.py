from src.experiment.utils import ParameterKeys
from abc import ABC, abstractmethod


class HPOptimizer(ABC):
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def objective(self):
        pass

    @abstractmethod
    def optimize(self):
        pass