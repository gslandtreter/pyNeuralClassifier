import neural_classifier
from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.neural_network import NeuralNetwork



if __name__ == '__main__':
    activation_function = ActivationFunction(neural_classifier.sigmoid, neural_classifier.derivative_sigmoid)

    topology = [1,4,2]

    neural_network = NeuralNetwork(topology, activation_function, alpha=0.15)

    error_data_points = []
    total_training_points = 0

    index = 0
    total_results = 0
    right_answers = 0

    inputs = [1]
    expected_output = [1, 0]

    inputs2 = [0]
    expected_output2 = [0, 1]

    for i in range(0, 1):
        # Training
        real_output = neural_network.evaluate(inputs)
        neural_network.backpropagate(expected_output)

        real_output2 = neural_network.evaluate(inputs2)
        neural_network.backpropagate(expected_output2)

    print real_output
