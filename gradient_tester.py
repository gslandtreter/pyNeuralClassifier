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

    for i in range(0, 100):
        # Training
        real_output = neural_network.evaluate(inputs)
        neural_network.backpropagate(expected_output, print_gradient_estimate=True)

    print "Expected Output:"
    print expected_output
    print "Real Output:"
    print real_output
