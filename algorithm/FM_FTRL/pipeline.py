#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: pipeline.py
@time: 2019-11-26 19:43
@desc:
"""

import os
import time
import math

import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from utils.logger import Logger
from utils.model import SecureFTRL
from utils.featurizer import set_seed
from utils.model_backup import SecureRegFTRL
from utils.simulation import DataGenerator, data_generator
from utils.parameter import params_global, params_experiment, params_ftrl_fm, params_reg_fm, params_lgb, columns


# set logger

if __name__ == "__main__":
    if not os.path.exists("modelResults"):
        os.mkdir("modelResults")
    # for sample_size in [int(5e3), int(5e4), int(5e5)]:
    sample_size = int(5e3)
    if not os.path.exists(f"modelResults/{sample_size}"):
        os.mkdir(f"modelResults/{sample_size}")
    for num_noisy_feature in [10, 20, 40]:
        for fm_dim in [3, 5, 8]:
            for exp_reg in [True, False]:
                for i in range(100):
                    filename = str(sample_size) + '_' + str(num_noisy_feature) + '_' + \
                               str(fm_dim) + '_' + str(exp_reg) + '_' + str(i)
                    logger = Logger(
                        filename="Simulation2019112521",
                        dir_name=f"logs/{time.strftime('%Y-%m-%d %H:%M')}",
                        console_mode=False
                    ).set_logger()
                    seed = int(abs(hash(filename))%10000)
                    set_seed(seed)
                    params_global["random_state"] = seed
                    params_experiment["sample_size"] = sample_size
                    params_experiment["num_noisy_feature"] = num_noisy_feature
                    params_ftrl_fm["fm_dim"] = fm_dim
                    params_reg_fm["fm_dim"] = fm_dim
                    params_reg_fm["exp_reg"] = exp_reg
                    for param in ["params_global",
                                  "params_experiment",
                                  "params_ftrl_fm",
                                  "params_reg_fm",
                                  "params_lgb"]:
                        logger.info(param)
                        logger.info(eval(param))
                    # generating the sample
                    logger.info("Generating the simulation data...")
                    data_simulator = DataGenerator(**params_experiment)
                    X, y = data_simulator.data_generate()
                    trn_X, val_X, trn_y, val_y = train_test_split(
                        X, y,
                        test_size=params_global["test_size"],
                        random_state=params_global["random_state"],
                        stratify=y
                    )
                    positive_instance_ratio = sum(y) / len(y)
                    logger.info(f"Number of 1s: {positive_instance_ratio}")

                    # training lr_ftrl
                    logger.info("Logistic Regression FTRL Training......")
                    params_ftrl_fm["interaction"] = False
                    secure_ftrl_lr = SecureFTRL(**params_ftrl_fm)
                    score_ftrl_lr = [0.] * len(trn_y)
                    ftrl_lr_auc_trn_100 = []
                    for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                        p = secure_ftrl_lr.predict(X)
                        score_ftrl_lr[i] = p
                        secure_ftrl_lr.update(X, p, y)
                        if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                            eval_metric = roc_auc_score(trn_y[:(i + 1)], score_ftrl_lr[:(i + 1)])
                            ftrl_lr_auc_trn_100.append(eval_metric)
                            print(i, eval_metric)
                    score_ftrl_lr = [0.] * len(val_y)
                    ftrl_lr_auc_val_100 = []
                    for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                        p = secure_ftrl_lr.predict(X)
                        score_ftrl_lr[i] = p
                        secure_ftrl_lr.update(X, p, y)
                        if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                            eval_metric = roc_auc_score(val_y[:(i + 1)], score_ftrl_lr[:(i + 1)])
                            ftrl_lr_auc_val_100.append(eval_metric)
                            print(i, eval_metric)
                    predictions = list(map(round, score_ftrl_lr))
                    logger.info("LR FTRL roc_auc_score")
                    ftrl_lr_roc_auc_score = roc_auc_score(val_y, score_ftrl_lr)
                    logger.info(ftrl_lr_roc_auc_score )
                    logger.info("LR FTRL f1_score")
                    ftrl_lr_f1_score = f1_score(val_y, predictions)
                    logger.info(ftrl_lr_f1_score)
                    logger.info("lR FTRL accuracy_score")
                    ftrl_lr_accuracy_score = accuracy_score(val_y, predictions)
                    logger.info(ftrl_lr_accuracy_score)

                    # training fm_ftrl
                    logger.info("Factorization Machine FTRL Training......")
                    params_ftrl_fm["interaction"] = True
                    secure_ftrl_fm = SecureFTRL(**params_ftrl_fm)
                    score_ftrl_fm = [0.] * len(trn_y)
                    ftrl_fm_auc_trn_100 = []
                    for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                        p = secure_ftrl_fm.predict(X)
                        score_ftrl_fm[i] = p
                        secure_ftrl_fm.update(X, p, y)
                        if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                            eval_metric = roc_auc_score(trn_y[:(i + 1)], score_ftrl_fm[:(i + 1)])
                            ftrl_fm_auc_trn_100.append(eval_metric)
                            print(i, eval_metric)
                    score_ftrl_fm = [0.] * len(val_y)
                    ftrl_fm_auc_val_100 = []
                    for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                        p = secure_ftrl_fm.predict(X)
                        score_ftrl_fm[i] = p
                        secure_ftrl_fm.update(X, p, y)
                        if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                            eval_metric = roc_auc_score(val_y[:(i + 1)], score_ftrl_fm[:(i + 1)])
                            ftrl_fm_auc_val_100.append(eval_metric)
                            print(i, eval_metric)
                    predictions = list(map(round, score_ftrl_fm))
                    logger.info("FM FTRL roc_auc_score")
                    ftrl_fm_roc_auc_score = roc_auc_score(val_y, score_ftrl_fm)
                    logger.info(ftrl_fm_roc_auc_score)
                    logger.info("FM FTRL f1_score")
                    ftrl_fm_f1_score = f1_score(val_y, predictions)
                    logger.info(ftrl_fm_f1_score)
                    logger.info("FM FTRL accuracy_score")
                    ftrl_fm_accuracy_score = accuracy_score(val_y, predictions)
                    logger.info(ftrl_fm_accuracy_score)

                    # training reg_fm_ftrl
                    logger.info("RegFM FTRL Training......")
                    secure_reg_fm = SecureRegFTRL(**params_reg_fm)
                    score_reg_fm = [0.] * len(trn_y)
                    reg_fm_auc_trn_100 = []
                    for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                        p = secure_reg_fm.predict(X)
                        score_reg_fm[i] = p
                        secure_reg_fm.update(X, p, y)
                        if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                            eval_metric = roc_auc_score(trn_y[:(i + 1)], score_reg_fm[:(i + 1)])
                            reg_fm_auc_trn_100.append(eval_metric)
                            print(i, eval_metric)
                    score_reg_fm = [0.] * len(val_y)
                    reg_fm_auc_val_100 = []
                    for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                        p = secure_reg_fm.predict(X)
                        score_reg_fm[i] = p
                        secure_reg_fm.update(X, p, y)
                        if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                            eval_metric = roc_auc_score(val_y[:(i + 1)], score_reg_fm[:(i + 1)])
                            reg_fm_auc_val_100.append(eval_metric)
                            print(i, eval_metric)
                    predictions = list(map(round, score_reg_fm))
                    logger.info("RegFM FTRL roc_auc_score")
                    reg_fm_roc_auc_score = roc_auc_score(val_y, score_reg_fm)
                    logger.info(reg_fm_roc_auc_score)
                    logger.info("RegFM FTRL f1_score")
                    reg_fm_f1_score = f1_score(val_y, predictions)
                    logger.info(reg_fm_f1_score)
                    logger.info("RegFM FTRL accuracy_score")
                    reg_fm_accuracy_score = accuracy_score(val_y, predictions)
                    logger.info(reg_fm_accuracy_score)

                    logger.info("LightGBM Training......")
                    lgb_trn = lgb.Dataset(
                        data=trn_X,
                        label=trn_y,
                    )
                    lgb_val = lgb.Dataset(
                        data=val_X,
                        label=val_y,
                        reference=lgb_trn
                    )
                    lgb_booster = lgb.train(
                        params=params_lgb,
                        train_set=lgb_trn,
                        num_boost_round=10000,
                        valid_sets=[lgb_trn, lgb_val],
                        early_stopping_rounds=200,
                        verbose_eval=100,
                        keep_training_booster=True
                    )
                    score_lgb = lgb_booster.predict(val_X)
                    predictions = list(map(round, score_lgb))
                    logger.info("lgb roc_auc_score")
                    lgb_roc_auc_score = roc_auc_score(val_y, score_lgb)
                    logger.info(lgb_roc_auc_score)
                    logger.info("lgb f1_score")
                    lgb_f1_score = f1_score(val_y, predictions)
                    logger.info(lgb_f1_score)
                    logger.info("lgb accuracy_score")
                    lgb_accuracy_score = accuracy_score(val_y, predictions)
                    logger.info(lgb_accuracy_score)

                    total_cut_valid = data_simulator.k_cut_valid
                    num_w_fst_simulator = len(data_simulator.w_fst_index)
                    num_w_snd_simulator = len(data_simulator.w_snd_index)
                    num_w_fst_lr = sum(_ != 0 for _ in secure_ftrl_lr.w)
                    num_w_fst_fm = sum(_ != 0 for _ in secure_ftrl_fm.w)
                    num_w_fst_reg_fm = sum(_ != 0 for _ in secure_reg_fm.w)

                    num_w_fst_lr_valid = len(
                        set(data_simulator.w_fst_index) &
                        set([i for i in range(int(params_ftrl_fm["D"])) if secure_ftrl_lr.w[i] != 0])
                    )
                    num_w_fst_fm_valid = len(
                        set(data_simulator.w_fst_index) &
                        set([i for i in range(int(params_ftrl_fm["D"])) if secure_ftrl_fm.w[i] != 0])
                    )
                    num_w_fst_reg_fm_valid = len(
                        set(data_simulator.w_fst_index) &
                        set([i for i in range(int(params_ftrl_fm["D"])) if secure_reg_fm.w[i] != 0])
                    )

                    w_snd_ftrl_fm = secure_ftrl_fm.w_fm.values()
                    w_snd_reg_fm = secure_reg_fm.w_fm.values()

                    num_w_snd_reg_fm = len(w_snd_ftrl_fm)
                    num_zeros_weight_w_snd_fm = sum(sum(_) == 0 for _ in w_snd_ftrl_fm)
                    num_zeros_weight_w_snd_reg_fm = sum(sum(_) == 0 for _ in w_snd_reg_fm)

                    num_zeros_w_snd_fm = sum(sum(_ == 0 for _ in weight) for weight in w_snd_ftrl_fm)
                    num_zeros_w_snd_reg_fm = sum(sum(_ == 0 for _ in weight) for weight in w_snd_reg_fm)

                    result = dict(zip(columns, [eval(_) for _ in columns]))
                    with open(f"modelResults/{sample_size}/{filename}", "wb") as f:
                        pickle.dump(result, f)
