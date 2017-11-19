import random


class Synapse(object):
    epsilon = 0.01

    def __init__(self, source, destination, alpha):
        self.source = source  # type: Neuron
        self.destination = destination  # type: Neuron

        self.weight = random.random() - 0.5
        self.alpha = alpha

        self.gradient = 0

    def process_gradient(self):
        self.gradient = self.source.output * self.destination.error
        return self.gradient

    def estimate_gradient(self):
        return ((self.source.output + self.epsilon) - self.source.activation_function.function(self.source.output - self.epsilon)) / 2 * self.epsilon

    def update_weight(self):
        self.weight -= self.alpha * self.gradient
