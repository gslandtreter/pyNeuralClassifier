from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.neural_network import NeuralNetwork
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def derivative_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


if __name__ == '__main__':

    activation_function = ActivationFunction(sigmoid, derivative_sigmoid)

    topology = [2, 3, 2]

    neural_network = NeuralNetwork(topology, activation_function, 0.1)

    inputs1 = [5, 1]
    expected_output1 = [1, 0]

    inputs2 = [1, 7]
    expected_output2 = [0, 1]

    for i in range(0, 1000):
        neural_network.evaluate(inputs1)
        neural_network.backpropagate(expected_output1)
        neural_network.evaluate(inputs2)
        neural_network.backpropagate(expected_output2)





    print neural_network.evaluate([6, 2])


