from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.layer import Layer
from neuralnetwork.neuron import Neuron
from neuralnetwork.synapse import Synapse

import math


class NeuralNetwork(object):
    def __init__(self, topology, activation_function, alpha):
        self.topology = topology
        self.activation_function = activation_function  # type: ActivationFunction
        self.layers = []  # type: list[Layer]
        self.mean_net_error = 0

        previous_layer = None  # type: Layer

        for layer_idx in topology:
            new_layer = Layer()

            for i in range(0, layer_idx):
                neuron = Neuron(activation_function)  # type: Neuron
                new_layer.add_neuron(neuron)

                if previous_layer is not None:
                    for partner in previous_layer.neurons:  # type: Neuron
                        new_synapse = Synapse(partner, neuron, alpha)  # type: Synapse
                        partner.output_synapses.append(new_synapse)
                        neuron.input_synapses.append(new_synapse)

            self.layers.append(new_layer)
            previous_layer = new_layer

    def evaluate(self, inputs):
        assert len(self.layers[0].neurons) == len(inputs)

        first_layer = self.layers[0]

        for i in range(0, len(first_layer.neurons)):
            first_layer.neurons[i].output = inputs[i]

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for neuron in layer.neurons:

                total_sum = 0

                for synapse in neuron.input_synapses:
                    total_sum += synapse.source.output * synapse.weight

                neuron.output = self.activation_function.function(total_sum)

        output = []

        for neuron in self.layers[-1].neurons:
            output.append(neuron.output)

        return output

    def backpropagate(self, expected_values):
        assert len(self.layers[-1].neurons) == len(expected_values)

        mean_net_error = 0

        for i in range(0, len(self.layers[-1].neurons)):
            added_error = expected_values[i] - self.layers[-1].neurons[i].output
            mean_net_error += added_error * added_error

        mean_net_error /= len(self.layers[-1].neurons)
        self.mean_net_error = math.sqrt(mean_net_error)

        for i in range(0, len(self.layers[-1].neurons)):
            neuron = self.layers[-1].neurons[i]
            neuron.process_output_gradient(expected_values[i])

        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]

            for neuron in layer.neurons:
                neuron.process_gradient()

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]

            for neuron in layer.neurons:
                for synapse in neuron.input_synapses:
                    synapse.update_weight()

    @staticmethod
    def get_output_class(probabilities):
        predicted_class = 1
        pc_prob = probabilities[0]

        for i in range(1, len(probabilities)):
            if probabilities[i] > pc_prob:
                pc_prob = probabilities[i]
                predicted_class = i + 1

        return predicted_class, pc_prob
