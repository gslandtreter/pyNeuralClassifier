import neural_classifier
from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.neural_network import NeuralNetwork


if __name__ == '__main__':
    activation_function = ActivationFunction(neural_classifier.sigmoid, neural_classifier.derivative_sigmoid)

    #Topologia simples, 1 neuronio entrada, 1 de saida e 1 na camada oculta
    topology = [1, 1, 1]

    #Inicializa rede neural sem neuronios de bias (para facilitar a verificacao do gradiente)
    neural_network = NeuralNetwork(topology, activation_function, alpha=0.15, has_bias_neurons=False)

    inputs = [1.5]
    expected_output = [1]

    ##########################################
    # Shumbay os valores dos pesos para bater com o exemplo dado. Por padrao, a rede inicializaria em random
    # Comentar para inicializacao randomica
    neural_network.layers[0].neurons[0].output_synapses[0].weight = 0.39
    neural_network.layers[1].neurons[0].output_synapses[0].weight = 0.94
    ##########################################

    print "Testing gradient calculation. Inputs:"
    print inputs

    print "Expected output:"
    print expected_output

    print "Topology:"
    print topology

    print "Evaluating... Weights:"
    weights = neural_network.get_weights()
    print weights

    print "Network output:"
    real_output = neural_network.evaluate(inputs)
    print real_output

    print "Network Cost (J):"
    cost = neural_network.get_cost(expected_output)
    print cost

    print "Running backpropagation..."
    neural_network.backpropagate(expected_output)

    print "Gradients:"
    gradients = neural_network.get_weight_gradients()
    print gradients

    epsilon = 0.000005
    print "Estimating gradients using epsilon = " + str(epsilon)
    gradient_estimates = neural_network.estimate_gradients(epsilon, inputs, expected_output)

    print "Estimated gradients:"
    print gradient_estimates

