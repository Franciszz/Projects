#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: main.py
@time: 2019-11-19 15:09
@desc:
"""

import os
import time
import math

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from utils.parameter import params_global, params_experiment, params_ftrl_fm, params_reg_fm
from utils.simulation import DataGenerator, data_generator
from utils.model import SecureFTRL
from utils.model_backup import SecureRegFTRL
from utils.logger import Logger

# set logger
marks = time.strftime("%Y-%m-%d %H:%M")
logger = Logger(filename="Simulation2019112521", dir_name=f"logs/{marks}").set_logger()


if __name__ == "__main__":
    for param in ["params_global", "params_experiment", "params_ftrl_fm", "params_reg_fm"]:
        logger.info(param + "\n")
        logger.info(eval(param))
        logger.info("Generating the simulation data...")
    data_simulator = DataGenerator(**params_experiment)
    X, y = data_simulator.data_generate()
    trn_X, val_X, trn_y, val_y = train_test_split(
        X, y,
        test_size=params_global["test_size"],
        random_state=params_global["random_state"],
        stratify=y
    )
    logger.info(f"Number of 1s: {sum(y)} / {len(y)}")

    logger.info("FTRL Training......")
    secure_ftrl_fm = SecureFTRL(**params_ftrl_fm)
    score_ftrl_fm = [0.] * len(trn_y)
    for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
        p = secure_ftrl_fm.predict(X)
        score_ftrl_fm[i] = p
        secure_ftrl_fm.update(X, p, y)
        if i > 0 and i % 1000 == 0:
            print(i, roc_auc_score(trn_y[:(i + 1)], score_ftrl_fm[:(i + 1)]))
    score_ftrl_fm = [0.] * len(val_y)
    for i, (X, y) in enumerate(data_generator(val_X, val_y)):
        p = secure_ftrl_fm.predict(X)
        score_ftrl_fm[i] = p
        secure_ftrl_fm.update(X, p, y)
        if i > 0 and i % 1000 == 0:
            print(i, roc_auc_score(val_y[:(i + 1)], score_ftrl_fm[:(i + 1)]))
    predictions = list(map(round, score_ftrl_fm))
    logger.info("FTRL roc_auc_score")
    logger.info(roc_auc_score(val_y, score_ftrl_fm))
    logger.info("FTRL f1_score")
    logger.info(f1_score(val_y, predictions))
    logger.info("FTRL accuracy_score")
    logger.info(accuracy_score(val_y, predictions))

    logger.info("Regularized FTRL Training......")
    secure_reg_fm = SecureRegFTRL(**params_reg_fm)
    score_reg_fm = [0.] * len(trn_y)
    for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
        p = secure_reg_fm.predict(X)
        score_reg_fm[i] = p
        secure_reg_fm.update(X, p, y)
        if i > 0 and i % 1000 == 0:
            print(i, roc_auc_score(trn_y[:(i + 1)], score_reg_fm[:(i + 1)]))
    score_reg_fm = [0.] * len(val_y)
    for i, (X, y) in enumerate(data_generator(val_X, val_y)):
        p = secure_reg_fm.predict(X)
        score_reg_fm[i] = p
        secure_reg_fm.update(X, p, y)
        if i > 0 and i % 1000 == 0:
            print(i, roc_auc_score(val_y[:(i + 1)], score_reg_fm[:(i + 1)]))
    predictions = list(map(round, score_reg_fm))
    logger.info("RegFTRL roc_auc_score")
    logger.info(roc_auc_score(val_y, score_reg_fm))
    logger.info("RegFTRL f1_score")
    logger.info(f1_score(val_y, predictions))
    logger.info("RegFTRL accuracy_score")
    logger.info(accuracy_score(val_y, predictions))
