import random

class Synapse(object):
    def __init__(self, source, destination, alpha):
        self.source = source  # type: Neuron
        self.destination = destination  # type: Neuron
        self.weight = random.random()
        self.delta_weight = 0
        self.alpha = alpha


    def update_weight(self):
        self.delta_weight = self.alpha * self.source.output * self.destination.gradient
        self.weight += self.delta_weight