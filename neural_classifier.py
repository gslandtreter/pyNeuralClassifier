from neuralnetwork.activation_function import ActivationFunction
from neuralnetwork.neural_network import NeuralNetwork
import math
import random
import csv

import matplotlib.pyplot as plt


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

    for i in range(0, len(normalized[0]) - 1):
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


def process_data(training_data, test_data, dataset_name, topology):
    activation_function = ActivationFunction(sigmoid, derivative_sigmoid)


    neural_network = NeuralNetwork(topology, activation_function, alpha=0.15)

    error_data_points = []
    total_training_points = 0

    with open(dataset_name + '_train.csv', 'wb') as csvtrain:
        fieldnames = ['instanceID', 'error']
        writer1 = csv.DictWriter(csvtrain, fieldnames=fieldnames)
        writer1.writeheader()

        with open(dataset_name + '_test.csv', 'wb') as csvtest:
            fieldnames = ['instanceID', 'expectedOutput', 'predictedOutput']
            writer2 = csv.DictWriter(csvtest, fieldnames=fieldnames)
            writer2.writeheader()

            index = 0
            total_results = 0
            right_answers = 0

            for (id, line) in enumerate(training_data):
                for i in range(0, 50):
                    index += 1
                    inputs = line[0:-1]
                    expected_output = int(line[-1])

                    output = build_expected_output(expected_output, topology[-1])

                    # Training
                    neural_network.evaluate(inputs)
                    neural_network.backpropagate(output)
                    neural_network.update_weights()

                    total_training_points += 1
                    if total_training_points < 50:
                        error_data_points.append(100 - neural_network.mean_net_error)

                    if index > 100:
                        index = 0
                        print "Treinando... Erro: {}".format(neural_network.mean_net_error)

                        writer1.writerow({'instanceID': id,
                                          'error': neural_network.mean_net_error})

            for (id, line) in enumerate(test_data):
                inputs = line[0:-1]
                expected_output = int(line[-1])

                real_output = neural_network.evaluate(inputs)
                real_output_class, prob = neural_network.get_output_class(real_output)

                writer2.writerow({'instanceID': id,
                                  'expectedOutput': expected_output,
                                  'predictedOutput': real_output_class})

                print real_output_class == expected_output

                total_results += 1
                if real_output_class == expected_output:
                    right_answers += 1

            print "Total de acertos: {}%".format(float(right_answers) / total_results * 100)

            plt.plot(error_data_points)
            plt.ylabel('Network Performance - ' + dataset_name)
            plt.xlabel('Number of presented samples')
            plt.show()

if __name__ == '__main__':

    # training_data, test_data = get_network_data("dataset/cmc.data", 0.8, False)
    # training_data, test_data = get_network_data("dataset/haberman.data", 0.8, False)

    dataset_name = "dataset/cmc.data"
    training_data, test_data = get_network_data(dataset_name, percentual_separation=0.8, class_first=False)
    topology = [9, 5, 3]
    process_data(training_data, test_data, dataset_name, topology)

    dataset_name = "dataset/haberman.data"
    training_data, test_data = get_network_data(dataset_name, percentual_separation=0.8, class_first=False)
    topology = [3, 5, 2]
    process_data(training_data, test_data, dataset_name, topology)

    dataset_name = "dataset/wine.data"
    training_data, test_data = get_network_data(dataset_name, percentual_separation=0.8, class_first=True)
    topology = [13, 5, 3]
    process_data(training_data, test_data, dataset_name, topology)


