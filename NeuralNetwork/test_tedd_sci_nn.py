import zipfile
import numpy as np

import tedd_sci_nn


n = tedd_sci_nn.NeuralNetwork(784, 100, 10, 0.3)
training_data_file = ''

with zipfile.ZipFile('../DataSets/mnist_dataset/mnist_train.zip', 'r') as z:
    training_data_file = z.open('mnist_train.csv', "r")


#training_data_file = open("../DataSets/mnist_dataset/mnist_train_100.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
    for record in training_data_list:
        record = record.decode("UTF-8")

        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        targets = np.zeros(10) + 0.01

        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        pass
    pass

#test_data_file = open("../DataSets/mnist_dataset/mnist_test_10.csv", "r")
test_data_file = open("../DataSets/mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print("Value = ", correct_label)

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print("Guess = ", label)

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("performance =", scorecard_array.sum()/scorecard_array.size)
