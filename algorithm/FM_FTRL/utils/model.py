#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: model.py
@time: 2019-11-18 21:17
@desc:
"""

from csv import DictReader
from math import exp, copysign, log, sqrt
from datetime import datetime
import random


class SecureFTRL(object):

    def __init__(self, L1, L2, alpha, beta, D,
                 L1_fm, L2_fm, fm_dim, fm_initDev,
                 alpha_fm, beta_fm, interaction, dropout_rate=1.0):
        """
        :param fm_dim: dimension of latent factorization machine weights.
        :param fm_initDev: standard deviation for random initialization of factorization weights.
        :param L1: L1 regularization for first order terms.
        :param L2: L2 regularization for second order terms.
        :param L1_fm: L1 regularization for factorization machine weights.
        :param L2_fm: L2 regularization for factorization machine weights.
        :param D: the dimension of parameter metric.
        :param alpha: learning rate parameter alpha.
        :param beta: learning rate parameter beta.
        :param alpha_fm: learning rate parameter alpha for factorization machine weights.
        :param beta_fm: learning rate parameter beta for factorization machine weights.
        :param dropout_rate: dropout rate for each variable
        :param interaction: logistic regression or factorization machine
        """

        self.fm_dim = fm_dim
        self.fm_initDev = fm_initDev
        self.L1 = L1
        self.L2 = L2
        self.L1_fm = L1_fm
        self.L2_fm = L2_fm
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.alpha_fm = alpha_fm
        self.beta_fm = beta_fm
        self.dropout_rate = 1 - dropout_rate
        self.interaction = interaction
        """
        model:
            n: squared sum of past gradients
            z: weights
            w: lazy weights
        """

        # set index 0 be bias term to avoid collisions
        self.n = [0.] * (D + 1)
        self.z = [0.] * (D + 1)
        self.w = [0.] * (D + 1)

        if self.interaction:
            self.n_fm = {}
            self.z_fm = {}
            self.w_fm = {}

    def init_fm(self, i):
        """
        Initialize the factorization machine weight vector for variable i.

        :param i: the variable index
        :return:
        """

        if i not in self.n_fm:
            self.n_fm[i] = [0.] * self.fm_dim
            self.w_fm[i] = [0.] * self.fm_dim
            self.z_fm[i] = [0.] * self.fm_dim

            for k in range(self.fm_dim):
                self.z_fm[i][k] = random.gauss(0., self.fm_initDev)

    def get_indices(self, x):
        """
        A helper generator that yields the indices in x
        The purpose of this generator is to make the following code cleaner when doing feature interaction

        :param x:
        :return:
        """
        for i, value in enumerate(x):
            yield int(abs(hash(str(i) + '_' + str(int(value))) % self.D))

    def get_indices_fm(self, x):
        """
        A helper generator that yields the innnteraction feature indices in x
        The purpose of this generator is to make the following code cleaner when doing feature interaction

        :param x:
        :return:
        """
        for i, value_i in enumerate(x[:-1]):
            for j, value_j in enumerate(x[1:]):
                yield int(hash(str(i) + '_' + str(value_i) + '_' + str(j) + '_' + str(value_j)) % self.D)

    def predict_raw(self, x):
        """
        predict the raw score prior to logistic transformation.

        :param x:
        :return:
        """
        #print("calculate raw score")
        indices = [_ for _ in self.get_indices(x)]
        #print(indices)
        raw_y = 0.
        # calculate the bias contribution
        for i in [0]:
            self.w[-1] = - self.z[i] / ((self.beta + sqrt(self.n[i])) / self.alpha)
            raw_y += self.w[-i]

        # calculate the first order contribution.
        for i in indices:
            # update the first order weights
            sign = -1. if self.z[i] < 0. else 1.
            if sign * self.z[i] <= self.L1:
                self.w[i] = 0.
            else:
                self.w[i] = (sign * self.L1 - self.z[i]) / ((self.beta + sqrt(self.n[i])) / self.alpha + self.L2)
            # add the first order contribution
            raw_y += self.w[i]

        # calculate factorization machine contribution.
        if self.interaction:
            # update the factorization machine weights
            for i in indices:
                self.init_fm(i)
                for k in range(self.fm_dim):
                    sign = -1. if self.z_fm[i][k] < 0. else 1.
                    if sign * self.z_fm[i][k] <= self.L1_fm:
                        self.w_fm[i][k] = 0.
                    else:
                        self.w_fm[i][k] = (sign * self.L1_fm - self.z_fm[i][k]) / (
                            (self.beta_fm + sqrt(self.n_fm[i][k])) / self.alpha_fm + self.L2_fm)
            # add the factorization machine weights
            for m in indices:
                for n in indices:
                    if n > m:
                        for k in range(self.fm_dim):
                            raw_y += self.w_fm[m][k] * self.w_fm[n][k]
        return raw_y

    def predict(self, x):
        """
        Calculate the logitic score
        :param self:
        :param x:
        :return:
        """
        #print("calculate prob")
        return 1. / (1. + exp(- max((min(self.predict_raw(x), 35.), -35.))))

    def update(self, x, p, y):
        """
        Update the Parameters Using Follow-the-Regularized-Leader Algorithm
        :param x:
        :param p:
        :param y:
        :return:
        """
        # print("updating")
        indices = [_ for _ in self.get_indices(x)]
        # cost gradient with respect to raw predictions
        g = p - y

        # sums for calculating gradients for factorization machine
        fm_sum = {}

        # update the weights for bias term
        for i in [0]:
            sigma = (sqrt(self.n[-1]) + g * g) - sqrt(self.n[-1]) / self.alpha
            self.z[-1] += g - sigma * self.w[-1]
            self.n[-1] += g * g

        # update the weights of the first order
        for i in indices:
            sigma = (sqrt(self.n[i]) + g * g) - sqrt(self.n[i]) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g
            fm_sum[i] = [0.] * self.fm_dim

        if self.interaction:
            # sum the gradients for factorization machine interaction weights
            for m in indices:
                for n in indices:
                    if n != m:
                        for k in range(self.fm_dim):
                            fm_sum[m][k] += self.w_fm[n][k]

            for i in indices:
                for k in range(self.fm_dim):
                    g_fm = g * fm_sum[i][k]
                    sigma = (sqrt(self.n_fm[i][k] + g_fm * g_fm) - sqrt(self.n_fm[i][k])) / self.alpha_fm
                    self.z_fm[i][k] += g_fm - sigma * self.w_fm[i][k]
                    self.n_fm[i][k] += g_fm * g_fm

    def dropout(self, x):
        """
        Dropout variable in list x
        :param x:
        :return:
        """
        pass

    def write_w(self, filename):
        """
        :param filename:
        :return:
        """
        with open(filename, 'w') as f_out:
            for i, w in enumerate(self.w):
                f_out.write("%i, %f\n" % (i, w))

    def write_w_fm(self, filename):
        """
        :param filename:
        :return:
        """
        with open(filename, "w") as f_out:
            for k, w_fm in self.w_fm.items():
                f_out.write("%i, %s\n" % (k, ",".join([str(w) for w in w_fm])))


def data_generator(x, y):
    for i in range(len(y)):
        yield x.iloc[i].values, y.iloc[i]

