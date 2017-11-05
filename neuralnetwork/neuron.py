from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.synapse import Synapse


class Neuron(object):

    def __init__(self, activ_func, is_bias_neuron):
        self.output = 0
        self.error = 0
        self.gradient = 0
        self.is_bias_neuron = is_bias_neuron

        self.activation_function = activ_func   # type: ActivationFunction
        self.input_synapses = []    # type: list[Synapse]
        self.output_synapses = []   # type: list[Synapse]

    def update_output(self):
        if self.is_bias_neuron:
            return

        input_sum = 0

        for synapse in self.input_synapses:
            input_sum += synapse.source.output * synapse.weight

        self.output = self.activation_function.function(input_sum)

    def process_error(self):
        total_sum = 0
        for synapse in self.output_synapses:
            total_sum += synapse.weight * synapse.destination.error

        self.error = total_sum * self.output * (1 - self.output)
        return self.error
