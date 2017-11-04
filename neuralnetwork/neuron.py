from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.synapse import Synapse


class Neuron(object):

    def __init__(self, activ_func):
        self.output = 0
        self.gradient = 0
        self.activation_function = activ_func   # type: ActivationFunction
        self.input_synapses = []    # type: list[Synapse]
        self.output_synapses = []   # type: list[Synapse]


    def process_output_gradient(self, expected_output):
        error = expected_output - self.output
        self.gradient = error * self.activation_function.derivative(self.output)


    def process_gradient(self):

        my_error_contribution = 0
        for synapse in self.output_synapses:
            my_error_contribution += synapse.destination.gradient * synapse.weight

        self.gradient = my_error_contribution * self.activation_function.derivative(self.output)

