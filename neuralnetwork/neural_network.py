from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.layer import Layer
from neuralnetwork.neuron import Neuron
from neuralnetwork.synapse import Synapse

import math


class NeuralNetwork(object):
    def __init__(self, topology, activation_function, alpha, has_bias_neurons=True):
        self.topology = topology
        self.activation_function = activation_function  # type: ActivationFunction
        self.layers = []  # type: list[Layer]
        self.mean_net_error = 1
        self.alpha = alpha
        self.output = []
        self.cost = 0
        self.has_bias_neurons = has_bias_neurons

        previous_layer = None  # type: Layer

        for layer_idx in range(0, len(topology)):
            new_layer = Layer()

            for i in range(0, topology[layer_idx]):
                neuron = Neuron(activation_function, is_bias_neuron=False)  # type: Neuron
                new_layer.add_neuron(neuron)

                if previous_layer is not None:
                    for partner in previous_layer.neurons:  # type: Neuron
                        new_synapse = Synapse(partner, neuron, alpha)  # type: Synapse
                        partner.output_synapses.append(new_synapse)
                        neuron.input_synapses.append(new_synapse)

            if layer_idx != len(topology) - 1 and has_bias_neurons:
                # Bias Neuron
                neuron = Neuron(activation_function, is_bias_neuron=True)  # type: Neuron
                neuron.output = 1
                new_layer.add_neuron(neuron)

            self.layers.append(new_layer)
            previous_layer = new_layer

    def evaluate(self, inputs):
        if self.has_bias_neurons:
            assert len(self.layers[0].neurons) - 1 == len(inputs)
        else:
            assert len(self.layers[0].neurons) == len(inputs)

        first_layer = self.layers[0]

        if self.has_bias_neurons:
            for i in range(0, len(first_layer.neurons) - 1):
                first_layer.neurons[i].output = inputs[i]
        else:
            for i in range(0, len(first_layer.neurons)):
                first_layer.neurons[i].output = inputs[i]

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for neuron in layer.neurons:
                neuron.update_output()

        self.output = []

        for neuron in self.layers[-1].neurons:
            self.output.append(neuron.output)

        return self.output

    def get_cost(self, expected_values):

        total_sum = 0
        for i in range(0, len(expected_values)):
            y = expected_values[i]
            fx = self.output[i]

            total_sum += (-y * math.log(fx)) - ((1 - y) * math.log(1 - fx))

        self.cost = total_sum / len(expected_values)
        return self.cost


    def backpropagate(self, expected_values):
        assert len(self.layers[-1].neurons) == len(expected_values)

        mean_net_error = 0

        # For each neuron in output layer
        # Calculates mean net error = sqrt(sum(error*error))
        for i in range(0, len(self.layers[-1].neurons)):
            added_error = self.layers[-1].neurons[i].output - expected_values[i]
            mean_net_error += added_error * added_error
            self.layers[-1].neurons[i].error = added_error

        mean_net_error /= len(self.layers[-1].neurons)
        self.mean_net_error = math.sqrt(mean_net_error)

        # For each layer, except output layer, backwards
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]

            for neuron in layer.neurons:
                neuron.process_error()

            for neuron in layer.neurons:
                for synapse in neuron.output_synapses:
                    synapse.process_gradient()

    def update_weights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for synapse in neuron.output_synapses:
                    synapse.update_weight()

    def get_weights(self):
        weights = []

        for layer in self.layers:
            for neuron in layer.neurons:
                for synapse in neuron.output_synapses:
                    weights.append(synapse.weight)

        return weights

    def get_weight_gradients(self):
        weights = []

        for layer in self.layers:
            for neuron in layer.neurons:
                for synapse in neuron.output_synapses:
                    weights.append(synapse.gradient)

        return weights

    def get_weight(self, index):
        curr_index = 0

        for layer in self.layers:
            for neuron in layer.neurons:
                for synapse in neuron.output_synapses:
                    if curr_index == index:
                        return synapse.weight
                    else:
                        curr_index += 1

        return None

    def set_weight(self, new_weight, index):
        curr_index = 0

        for layer in self.layers:
            for neuron in layer.neurons:
                for synapse in neuron.output_synapses:
                    if curr_index == index:
                        synapse.weight = new_weight
                        return new_weight
                    else:
                        curr_index += 1

        return None

    def estimate_gradients(self, epsilon, inputs, expected_output):
        estimated_gradients = []

        for i in range(0, len(self.get_weights())):

            original_weight = self.get_weight(i)

            # J(w1 + epsilon)
            self.set_weight(original_weight + epsilon, i)
            self.evaluate(inputs)
            j_pos = self.get_cost(expected_output)

            # J(w1 - epsilon)
            self.set_weight(original_weight - epsilon, i)
            self.evaluate(inputs)
            j_neg = self.get_cost(expected_output)

            estimated_gradient = (j_pos - j_neg) / (2 * epsilon)
            estimated_gradients.append(estimated_gradient)

            # Reset original weight
            self.set_weight(original_weight, i)

        return estimated_gradients



    @staticmethod
    def get_output_class(probabilities):
        predicted_class = 1
        pc_prob = probabilities[0]

        for i in range(1, len(probabilities)):
            if probabilities[i] > pc_prob:
                pc_prob = probabilities[i]
                predicted_class = i + 1

        return predicted_class, pc_prob
