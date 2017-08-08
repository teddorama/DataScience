import numpy as np
import scipy.special
from matplotlib import pyplot as plt


class NeuralNetwork:
    """ Neural Network
    """

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                    (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),
                                    (self.onodes, self.hnodes))

        self.inputs = None
        self.targets = None
        self.hidden_inputs = None
        self.hidden_outputs = None
        self.final_inputs = None
        self.final_outputs = None

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        self.targets = np.array(targets_list, ndmin=2).T
        self.query(inputs_list)

        output_errors = self.targets - self.final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr \
                    * np.dot(
                            output_errors * self.final_outputs 
                            * (1.0-self.final_outputs)
                            , np.transpose(self.hidden_outputs)
                            )
        self.wih += self.lr * np.dot(hidden_errors * self.hidden_outputs * (1.0-self.hidden_outputs), np.transpose(self.inputs))

        pass

    def query(self, inputs_list):
        self.inputs = np.array(inputs_list, ndmin=2).T

        self.hidden_inputs = np.dot(self.wih, self.inputs)
        self.hidden_outputs = self.activation_function(self.hidden_inputs)

        self.final_inputs = np.dot(self.who, self.hidden_outputs)
        self.final_outputs = self.activation_function(self.final_inputs)

        return self.final_outputs


if __name__ == "__main__":
    n = NeuralNetwork(3, 3, 3, 0.3)
    print(n.wih)
    print(n.query([1.0, 0.5, -1.5]))

    data_file = open("../DataSets/mnist_dataset/mnist_train_100.csv", "r")
    data_list = data_file.readlines()
    data_file.close()

    all_values = data_list[0].split(',')
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
