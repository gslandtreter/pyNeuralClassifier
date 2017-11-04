from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.neural_network import NeuralNetwork
import math

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

def derivative_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def normalize(content):
    normalized = []

    for line in content:
        tokens = line.split(",")
        new_token = []
        for x in tokens:
            new_token.append(float(x))
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



if __name__ == '__main__':

    activation_function = ActivationFunction(sigmoid, derivative_sigmoid)

    topology = [3, 10, 2]

    neural_network = NeuralNetwork(topology, activation_function, 0.1)

    with open("dataset/haberman.data.txt") as f:
        content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    index = 0
    total_results = 0
    right_answers = 0

    content = normalize(content)

    separation = int(math.floor(len(content) * 1))
    training_data = content[0: separation]
    test_data = content[separation: len(content)]

    for i in range(0, 1000):
        for line in training_data:
            index += 1
            inputs = [line[0], line[1], line[2]]
            expected_output = int(line[3])

            if expected_output == 1:
                output = [1, 0]
            else:
                output = [0, 1]

            # Training
            neural_network.evaluate(inputs)
            neural_network.backpropagate(output)

            if index > 100:
                index = 0
                print neural_network.mean_net_error




    for line in content:
        inputs = [line[0], line[1], line[2]]
        expected_output = int(line[3])

        real_output = neural_network.evaluate(inputs)
        real_output_class, prob = neural_network.get_output_class(real_output)

        print real_output_class == expected_output

        total_results += 1
        if real_output_class == expected_output:
            right_answers += 1

    print float(right_answers) / total_results * 100










