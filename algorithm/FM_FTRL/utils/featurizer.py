#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: featurizer.py
@time: 2019-11-19 15:10
@desc:
"""

import random
import pandas as pd
import numpy as np


def qcut(feature, k):
    return pd.cut(feature, bins=k, labels=range(k))


def set_seed(random_state):
    np.random.seed(random_state)
    random.seed(random_state)
