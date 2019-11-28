#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: simulation.py
@time: 2019-11-25 19:12
@desc:
"""

import pandas as pd
import numpy as np
import random
from math import exp


class DataGenerator:

    def __init__(self, sample_size, D,
                 prob_threshold, w_fst_threshold, w_snd_threshold,
                 num_valid_feature, num_noisy_feature, num_valid_feature_ratio,
                 std_init_weight_fst, std_init_weight_snd,
                 k_cut_tuple_n=(2, 5, 10, 20, 50),
                 k_cut_tuple_p=(0.3, 0.2, 0.2, 0.2, 0.1)):
        """
        :param sample_size:
        :param D:
        :param threshold:
        :param num_valid_feature:
        :param num_noisy_feature:
        :param std_init_weight_fst:
        :param std_init_weight_snd:
        :param k_cut_tuple_n:
        :param k_cut_tuple_p:
        """
        self.sample_size = sample_size
        self.D = D
        self.prob_threshold = prob_threshold
        self.w_fst_threshold = w_fst_threshold
        self.w_snd_threshold = w_snd_threshold
        self.num_valid_feature = num_valid_feature
        self.num_noisy_feature = num_noisy_feature * num_valid_feature
        self.num_valid_related_ratio = int(num_valid_feature_ratio * num_valid_feature)
        self.std_init_weight_fst = std_init_weight_fst
        self.std_init_weight_snd = std_init_weight_snd
        self.k_cut_tuple_n = k_cut_tuple_n
        self.k_cut_tuple_p = k_cut_tuple_p

    def get_indices(self, x):
        """
        :param x:
        :return:
        """
        for i, value in enumerate(x):
            yield int(abs(hash(str(i) + '_' + str(int(value))) % self.D))

    def get_indices_fm(self, x):
        """
        :param x:
        :return:
        """
        for i, value_i in enumerate(x):
            for j, value_j in enumerate(x):
                if j > i:
                    yield int(abs(
                        hash(str(i) + '_' + str(value_i) + '_' + str(j) + '_' + str(value_j)) % self.D
                    ))

    def data_generate(self):
        self.k_cut_valid = np.random.choice(
            self.k_cut_tuple_n,
            size=self.num_valid_feature,
            p=self.k_cut_tuple_p
        )
        self.k_cut_noisy = np.random.choice(
            self.k_cut_tuple_n,
            size=self.num_noisy_feature,
            p=self.k_cut_tuple_p
        )
        self.w_all_index = [0.] * self.D
        self.w_fst_index = []
        self.w_snd_index = []
        self.w_snd_related_index = set()
        self.logits = [0.] * self.sample_size
        self.labels = [0.] * self.sample_size
        valid_X = np.random.multivariate_normal(
            mean=[0.]*self.num_valid_feature,
            cov=np.eye(self.num_valid_feature),
            size=self.sample_size
        )
        noisy_X = np.random.multivariate_normal(
            mean=[0.]*self.num_noisy_feature,
            cov=np.eye(self.num_noisy_feature),
            size=self.sample_size
        )
        for i, kcut in enumerate(self.k_cut_valid):
            valid_X[:, i] = pd.cut(valid_X[:, i], bins=kcut, labels=list(map(int, range(kcut))))
        for i, kcut in enumerate(self.k_cut_noisy):
            noisy_X[:, i] = pd.cut(noisy_X[:, i], bins=kcut, labels=list(map(int, range(kcut))))
        for i in range(self.num_valid_feature):
            for value_i in range(self.k_cut_valid[i]):
                index = int(abs(hash(str(i) + '_' + str(value_i)) % self.D))
                weight = random.normalvariate(0, self.std_init_weight_fst)
                if abs(weight) > self.w_fst_threshold:
                    self.w_fst_index.append(index)
                    self.w_all_index[index] = weight
        for i, i_cut in enumerate(self.k_cut_valid[:self.num_valid_related_ratio]):
            for j, j_cut in enumerate(self.k_cut_valid[:self.num_valid_related_ratio]):
                if j > i:
                    for value_i in range(i_cut):
                        for value_j in range(j_cut):
                            weight = random.normalvariate(0, self.std_init_weight_snd)
                            if random.random() > self.prob_threshold and abs(weight) > self.w_snd_threshold:
                                hash_i = abs(hash(str(i) + '_' + str(int(value_i))) % self.D)
                                hash_j = abs(hash(str(j) + '_' + str(int(value_j))) % self.D)
                                if hash_i in self.w_fst_index and hash_j in self.w_fst_index:
                                    self.w_snd_related_index.add(hash_i)
                                    self.w_snd_related_index.add(hash_j)
                                    index = int(abs(hash(str(hash_i) + '_' + str(hash_j)) % self.D))
                                    # print(i, value_i, j, value_j, hash_i, hash_j, index)
                                    # index = int(abs(
                                    #     hash(str(i) + '_' + str(value_i) + '_' + str(j) + '_' + str(value_j)) % self.D
                                    # ))
                                    self.w_snd_index.append(index)
                                    self.w_all_index[index] = weight
        for i, row in enumerate(valid_X):
            score = random.uniform(-1, 1)
            for j in self.get_indices(row):
                score += self.w_all_index[j]
            for j in self.get_indices(row):
                score += self.w_all_index[j]
            logit = 1. / (1. + exp(- max((min(score, 35.), -35.))))
            self.logits[i] = logit
            if random.random() < logit:
                self.labels[i] = 1.
        return np.c_[valid_X, noisy_X], self.labels


def data_generator(X, y):
    for i in range(len(y)):
        yield X[i, :], y[i]






