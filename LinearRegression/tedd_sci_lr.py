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

        self._x = self.sample.data[:, i]
        self._x_sklearn = self.sample.data[:, np.newaxis, i]     #diabetes.data[:, 2].reshape(-1, 1)
        self._x_train = self._x[:-test_scope]
        self._x_test = self._x[-test_scope:]

        self._y = self.sample.target[:]
        self._y_train = self._y[:-test_scope]
        self._y_test = self._y[-test_scope:]

        self._x_len = len(self._x)

    def lr_standard(self):
        regr = linear_model.LinearRegression()
        regr.fit(self._x_sklearn, self._y)

        _y_pred = regr.predict(self._x_sklearn)

        self.show_graph(self._x_sklearn, _y_pred
                        , "SK Learn - Linigar Regression"
                        , regr.coef_[0], regr.intercept_)

    def lr_sgd(self):
        regr = linear_model.SGDRegressor(n_iter=30000, penalty='none')
        regr.fit(self._x_sklearn, self._y)

        _y_pred = regr.predict(self._x_sklearn)

        self.show_graph(self._x_sklearn, _y_pred
                        , "SK Learn - SGD Linigar Regression"
                        , regr.coef_[0], regr.intercept_[0])

    def lr_hacker(self, w=1, b=1, n=3000, lr_rate=0.1):
        _w = []
        _b = []

        for i in range(n):
            y_hat = w*self._x + b
            err = self._y - y_hat

            w += lr_rate * np.sum(err * self._x) / self._x_len
            b += lr_rate * np.sum(err * 1) / self._x_len
            _w.append(w)
            _b.append(b)

        _y_pred = w*self._x + b

        self.show_graph(self._x, _y_pred
                        , "Hacker - Linigar Regression"
                        , w, b)
        self.show_coef(n, _w, _b)

    def lr_tedd_1(self, w=1, b=1, n=3000, lr_rate=0.1):
        _w = []
        _b = []

        for i in range(n):
            y_hat = w*self._x + b
            err = self._y - y_hat

            w += lr_rate * np.sum(err / self._x) / self._x_len
            b += lr_rate * np.sum(err * 1) / self._x_len
            _w.append(w)
            _b.append(b)

        _y_pred = w*self._x + b

        self.show_graph(self._x, _y_pred
                        , "TEDD #1 - Linigar Regression"
                        , w, b)
        self.show_coef(n, _w, _b)

    def lr_tedd_2(self, w=1, b=1, n=3000, lr_rate=0.1):
        _w = []
        _b = []

        for i in range(n):
            y_hat = w*self._x + b
            err = self._y - y_hat

            w += lr_rate * np.sum(err) / self._x_len
            b += lr_rate * np.sum(err * 1) / self._x_len * 0.3
            _w.append(w)
            _b.append(b)

        _y_pred = w*self._x + b

        self.show_graph(self._x, _y_pred
                        , "TEDD #2 - Linigar Regression"
                        , w, b)
        self.show_coef(n, _w, _b)

    def lr_tedd_3(self, w=1, b=1, n=3000, lr_rate=0.1):
        _w = []
        _b = []

        for i in range(n):
            y_hat = self._get_pred(w, b)
            err = self._y - y_hat

            dw = lr_rate * np.sum(err/self._x) / self._x_len

            if self._get_cost(self._get_pred(w+dw, b)) <= self._get_cost(self._get_pred(w-dw, b)):
                w += dw
            else:
                w -= dw

            db = lr_rate * np.sum(err * 1) / self._x_len

            if self._get_cost(self._get_pred(w, b+db)) <= self._get_cost(self._get_pred(w, b-db)):
                b += db
            else:
                b -= db

            _w.append(w)
            _b.append(b)

        _y_pred =  self._get_pred(w, b)

        self.show_graph(self._x, _y_pred
                        , "TEDD #3 - Linigar Regression"
                        , w, b)
        self.show_coef(n, _w, _b)

    def lr_tedd_4(self, w=1, b=1, n=3000, lr_rate=0.1):
        #@TODO : 초기 할당값을 약간 크게 흔들다가 점차 logn으로 작게 수렴하게 해보자
        _w = []
        _b = []

        for i in range(n):
            y_hat = self._get_pred(w, b)
            err = self._y - y_hat

            dw = lr_rate * np.sum(err/self._x) / self._x_len
            w = self._get_opt_weight(w, b, dw, 10)

            db = lr_rate * np.sum(err * 1) / self._x_len
            b = self._get_opt_bias(w, b, db)

            _w.append(w)
            _b.append(b)

        _y_pred =  self._get_pred(w, b)

        self.show_graph(self._x, _y_pred, "TEDD #4 - Linigar Regression"
                        , w, b)
        self.show_coef(n, _w, _b)

    def _get_cost(self, y):
        return np.mean((self._y - y) ** 2)

    def _get_pred(self, w, b):
        return w*self._x + b

    def _get_opt_weight(self, w, b, dw, cnt):
        cost = self._get_cost(self._get_pred(w, b))
        cost_dw_plus = self._get_cost(self._get_pred(w+dw, b))
        cost_dw_minus = self._get_cost(self._get_pred(w-dw, b))

        if (cnt == 0):
            return w

        cnt -= 1

        if cost <= cost_dw_plus and cost <= cost_dw_minus:
            return self._get_opt_weight(w, b, dw/2, cnt)
        elif cost_dw_plus <= cost_dw_minus:
            return self._get_opt_weight(w+dw, b, dw/2, cnt)
        else:
            return self._get_opt_weight(w-dw, b, dw/2, cnt)

    def _get_opt_bias(self, w, b, db):
        cost = self._get_cost(self._get_pred(w, b))
        cost_db_plus = self._get_cost(self._get_pred(w, b+db))
        cost_db_minus = self._get_cost(self._get_pred(w, b-db))

        if cost <= cost_db_plus and cost <= cost_db_minus:
            return b
        elif cost_db_plus <= cost_db_minus:
#            return self._get_opt_weight(w, b+db, db/2)
            return b+db
        else:
#            return self._get_opt_weight(w, b-db, db/2)
            return b-db

    def show_graph(self, x=[], y=[], label='', w=0, b=0):
        print("#############################################################")
        print(label)
        print("Weight = {:.2f}  ,Bias = {:.2f}".format(w, b))
        print("Mean squared error = {:.2f}".format(np.mean((y - self._y) ** 2)))

        plt.scatter(self._x_train, self._y_train)
        plt.scatter(self._x_test, self._y_test, color='black')
        plt.plot(x, y, color='blue', linewidth=3)
        plt.show()

    def show_coef(self, n, w, b):
        plt.plot(range(n), w, linewidth=1)
        plt.plot(range(n), b, linewidth=1)
        plt.show()

        print(w[:5])
        print(b[:5])
        print("#############################################################")

lr = LinearRegression(datasets.load_diabetes(), 2)
#lr = LinearRegression(datasets.load_boston(), 12)
lr.lr_standard()
lr.lr_sgd()
lr.lr_hacker(5, 1, 7000)
lr.lr_tedd_3(1, 1, 7000, 0.1)
lr.lr_tedd_4(1, 1, 300, 1)





