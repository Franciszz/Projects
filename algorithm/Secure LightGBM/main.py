#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: main.py
@time: 2019-11-17 14:01
@desc:
"""

import os
x = range(5)
D = 1e10


def generator(x):
    for i in x:
        yield i


# for m, value_m in enumerate(x[:-1]):
#     for n, value_n in enumerate(x[1:]):
#         print(hash(str(m) + '_' + str(value_m) + '_' + str(n) + '_' + str(value_n)) % D)
# del x, D, m, value_m, value_n, n
for i in generator(x):
    for j in generator(x):
        print(i, j)