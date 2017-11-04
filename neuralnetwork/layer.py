
class Layer(object):

    def __init__(self):
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def get_size(self):
        return len(self.neurons)