from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.neural_network import NeuralNetwork
import math
import random

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

def derivative_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def normalize(content, class_first):
    normalized = []

    for line in content:
        tokens = line.split(",")
        new_token = []

        for x in tokens:
            new_token.append(float(x))

        if class_first:
            new_token[0], new_token[-1] = new_token[-1], new_token[0]

        normalized.append(new_token)

    for i in range (0, len(normalized[0]) - 1):
        min = 99999999
        max = -99999999

        for j in range(0, len(normalized)):
            if normalized[j][i] < min:
                min = normalized[j][i]
            if normalized[j][i] > max:
                max = normalized[j][i]

        for j in range(0, len(normalized)):
           normalized[j][i] = (normalized[j][i] - min) / (max - min)

    return normalized


def get_network_data(file_name, percentual_separation, class_first):
    with open(file_name) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    content = normalize(content, class_first)
    random.shuffle(content)

    separation = int(math.floor(len(content) * percentual_separation))
    training_data = content[0: separation]
    test_data = content[separation: len(content)]

    return training_data, test_data

def build_expected_output(class_id, dimension):
    output = [0] * dimension
    output[class_id - 1] = 1
    return output

if __name__ == '__main__':

    activation_function = ActivationFunction(sigmoid, derivative_sigmoid)

    input_layer_size = 3
    output_layer_size = 2

    topology = [input_layer_size, 4, output_layer_size]
    neural_network = NeuralNetwork(topology, activation_function, 0.1)

    #training_data, test_data = get_network_data("dataset/cmc.data", 0.8, False)
    training_data, test_data = get_network_data("dataset/haberman.data", 0.8, False)
    #training_data, test_data = get_network_data("dataset/wine.data", 0.8, True)


    index = 0
    total_results = 0
    right_answers = 0



    for line in training_data:
        for i in range(0, 10):
            index += 1
            inputs = line[0:-1]
            expected_output = int(line[-1])

            output = build_expected_output(expected_output, output_layer_size)

            # Training
            neural_network.evaluate(inputs)
            neural_network.backpropagate(output)

            if index > 100:
                index = 0
                print neural_network.mean_net_error

    for line in test_data:
        inputs = line[0:-1]
        expected_output = int(line[-1])

        real_output = neural_network.evaluate(inputs)
        real_output_class, prob = neural_network.get_output_class(real_output)

        print real_output_class == expected_output

        total_results += 1
        if real_output_class == expected_output:
            right_answers += 1

    print "Total de acertos: {}%".format(float(right_answers) / total_results * 100)










