import numpy as np
from abc import ABC, abstractmethod
from activation_functions import ActivationFunction

# Katmanlar için arayüz sınıfı
class Layer(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, output_error, learning_rate):
        pass

class DenseLayer(Layer):
    def __init__(self, output_size, activation_function, input_size=None):
        self.output_size = output_size
        self.activation_function = activation_function()
        self.input_size = input_size
        self.weights = None
        self.biases = None

    def initialize(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size, self.output_size)
        self.biases = np.zeros((1, self.output_size))

    def forward(self, inputs):
        if self.weights is None:
            self.initialize(inputs.shape[1])

        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function(self.z)
        return self.a

    def backward(self, output_error, learning_rate):
        d_activation = output_error * self.activation_function.derivative(self.a)
        input_error = np.dot(d_activation, self.weights.T)

        # Weight and bias updates
        self.weights += learning_rate * np.dot(self.inputs.T, d_activation)
        self.biases += learning_rate * np.sum(d_activation, axis=0, keepdims=True)

        return input_error
