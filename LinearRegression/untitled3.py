# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 00:52:29 2017

@author: teddo
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../NeuralNetwork")
import tedd_sci_nn


###############################################################################
# To Be Deleted
###############################################################################

n = tedd_sci_nn.NeuralNetwork(3, 3, 3, 0.3)
print(n.wih)
print(n.query([1.0, 0.5, -1.5]))

data_file = open("../DataSets/mnist_dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')