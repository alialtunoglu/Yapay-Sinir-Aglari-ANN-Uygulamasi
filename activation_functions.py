import numpy as np
from abc import ABC, abstractmethod

# Transfer fonksiyonları için arayüz sınıfı
class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)
