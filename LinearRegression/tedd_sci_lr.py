# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:51:20 2017

@author: teddo
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model


print(__doc__)


class LinearRegression:
    def __init__(self, data, i):
        self.sample = data
        test_scope = 20
        
        self.sample_x = self.sample.data[:, np.newaxis, i]
        self.sample_x_train = self.sample_x[:-test_scope]
        self.sample_x_test = self.sample_x[-test_scope:]
        
        self.sample_y = self.sample.target[:]
        self.sample_y_train = self.sample_y[:-test_scope]
        self.sample_y_test = self.sample_y[-test_scope:]

    
    def lr_standard(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.sample_x, self.sample_y)
        
        self.show_graph(self.sample_x, regr.predict(self.sample_x)
                        , "SK Learn - Linigar Regression"
                        , regr.coef_[0], regr.intercept_)
        
    def lr_sgd(self):
        regr = linear_model.SGDRegressor(n_iter=30000, penalty='none')
        #sgd_regr.fit(diabetes.data[:, 2].reshape(-1, 1), diabetes.target)
        regr.fit(self.sample_x, self.sample_y)
        
        self.show_graph(self.sample_x, regr.predict(self.sample_x)
                        , "SK Learn - SGD Linigar Regression"
                        , regr.coef_[0], regr.intercept_[0])
        
    def lr_hacker(self, w=1, b=1, n=3000):
        self._x = self.sample.data[:, 2]
        m = len(self._x)

        for i in range(n):
            y_hat = w*self._x + b
            err = self.sample_y - y_hat
            
            w += 0.1 * np.sum(err * self._x) / m
            b += 0.1 * np.sum(err * 1) / m
            #print(w,b)

        y_pred = w*self._x + b
        
        self.show_graph(self._x, y_pred
                        , "Hacker - Linigar Regression"
                        , w, b)
        
    def lr_tedd_1(self):
        self._x = self.sample.data[:, 2]
        m = len(self._x)
        
        #x1 = np.mean
        
    def show_graph(self, x=[], y=[], label='', w=0, b=0):
        print("#############################################################")
        print(label)
        print("Weight = {:.2f}  ,Bias = {:.2f}".format(w, b))
        print("Mean squared error = {:.2f}".format(np.mean((y - self.sample_y) ** 2)))

        plt.scatter(self.sample_x_train, self.sample_y_train)
        plt.scatter(self.sample_x_test, self.sample_y_test, color='black')
        plt.plot(x, y, color='blue', linewidth=3)
        plt.show()

        print("#############################################################")

lr = LinearRegression(datasets.load_diabetes(), 2)
#lr = LinearRegression(datasets.load_boston(), 12)
lr.lr_standard()
lr.lr_sgd()
lr.lr_hacker(5,1, 30000)

 


