from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.synapse import Synapse


class Neuron(object):

    def __init__(self, activ_func):
        self.total_input = 0
        self.output = 0
        self.gradient = 0
        self.output_der = 0
        self.bias = 0.1
        self.input_der = 0
        self.acc_input_der = 0
        self.num_acc_ders = 0

        self.activation_function = activ_func   # type: ActivationFunction
        self.input_synapses = []    # type: list[Synapse]
        self.output_synapses = []   # type: list[Synapse]


    def update_output(self):
        self.total_input = self.bias

        for synapse in self.input_synapses:
            self.total_input += synapse.source.output * synapse.weight

        self.output = self.activation_function.function(self.total_input)

    def process_gradient(self):
        self.input_der = self.output_der * self.activation_function.derivative(self.total_input)
        self.acc_input_der += self.input_der
        self.num_acc_ders += 1

