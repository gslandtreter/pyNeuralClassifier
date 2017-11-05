import random

class Synapse(object):
    def __init__(self, source, destination, alpha):
        self.source = source  # type: Neuron
        self.destination = destination  # type: Neuron
        self.weight = random.random() - 0.5
        self.delta_weight = 0
        self.alpha = alpha

        self.error_der = 0
        self.acc_error_der = 0
        self.num_acc_ders = 0


    def process_error(self):
        self.error_der = self.destination.input_der * self.source.output
        self.acc_error_der += self.error_der
        self.num_acc_ders += 1

    def update_weight(self):
        self.delta_weight = self.alpha * self.source.output * self.destination.gradient
        self.weight += self.delta_weight