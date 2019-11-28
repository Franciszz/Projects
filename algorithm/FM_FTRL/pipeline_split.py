#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: pipeline_split.py
@time: 2019-11-26 21:30
@desc:
"""

import os
import time
import math

import numpy as np
import pandas as pd

import pickle
import tqdm
# import lightgbm as lgb

from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from utils.logger import Logger
from utils.model import SecureFTRL
from utils.featurizer import set_seed
from utils.model_backup import SecureRegFTRL
from utils.simulation import DataGenerator, data_generator
from utils.parameter import params_global, params_experiment, params_ftrl_fm, params_reg_fm, columns, columns_display

if __name__ == "__main__":
    if not os.path.exists("modelResults"):
        os.mkdir("modelResults")
    # for sample_size in [int(5e3), int(5e4), int(5e5)]:
    sample_size = int(1e4)
    if not os.path.exists(f"modelResults/{sample_size}"):
        os.mkdir(f"modelResults/{sample_size}")
    for num_noisy_feature in tqdm.tqdm([1, 2, 4]):
        for fm_dim in tqdm.tqdm([3, 5, 7]):
            for i in tqdm.tqdm(range(100)):
                filename = str(sample_size) + '_' + str(num_noisy_feature) + '_' + str(fm_dim) + '_' + str(i)
                if os.path.exists(f"modelResults/{sample_size}/{filename}"):
                    pass
                else:
                    logger = Logger(
                        filename="Simulation2019112521",
                        dir_name=f"logs/{time.strftime('%Y-%m-%d %H:%M')}",
                        console_mode=False
                    ).set_logger()
                    seed = int(abs(hash(filename)) % 10000)
                    set_seed(seed)
                    params_global["random_state"] = seed
                    params_experiment["sample_size"] = sample_size
                    params_experiment["num_noisy_feature"] = num_noisy_feature
                    params_ftrl_fm["fm_dim"] = fm_dim
                    params_reg_fm["fm_dim"] = fm_dim
                    for param in ["params_global",
                                  "params_experiment",
                                  "params_ftrl_fm",
                                  "params_reg_fm"]:
                        logger.info(param)
                        logger.info(eval(param))

                    # generating the sample
                    logger.info("Generating the simulation data...")
                    data_simulator = DataGenerator(**params_experiment)
                    X, y = data_simulator.data_generate()
                    positive_instance_ratio = sum(y) / len(y)
                    if positive_instance_ratio > 0.95 or positive_instance_ratio < 0.05:
                        pass
                    else:
                        trn_X, val_X, trn_y, val_y = train_test_split(
                            X, y,
                            test_size=params_global["test_size"],
                            random_state=params_global["random_state"],
                            stratify=y
                        )

                        logger.info(f"Number of 1s: {positive_instance_ratio}")

                        # training lr_ftrl
                        logger.info("Logistic Regression FTRL Training......")
                        params_ftrl_fm["interaction"] = False
                        secure_ftrl_lr = SecureFTRL(**params_ftrl_fm)
                        score_ftrl_lr = [0.] * len(trn_y)
                        auc_trn_100_ftrl_lr = []
                        for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                            p = secure_ftrl_lr.predict(X)
                            score_ftrl_lr[i] = p
                            secure_ftrl_lr.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                                eval_metric = roc_auc_score(trn_y[:(i + 1)], score_ftrl_lr[:(i + 1)])
                                auc_trn_100_ftrl_lr.append(eval_metric)
                        score_ftrl_lr = [0.] * len(val_y)
                        auc_val_100_ftrl_lr = []
                        for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                            p = secure_ftrl_lr.predict(X)
                            score_ftrl_lr[i] = p
                            secure_ftrl_lr.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                                eval_metric = roc_auc_score(val_y[:(i + 1)], score_ftrl_lr[:(i + 1)])
                                auc_trn_100_ftrl_lr.append(eval_metric)
                        predictions = list(map(round, score_ftrl_lr))
                        logger.info("LR FTRL roc_auc_score")
                        roc_auc_score_ftrl_lr = roc_auc_score(val_y, score_ftrl_lr)
                        logger.info(roc_auc_score_ftrl_lr)
                        logger.info("LR FTRL f1_score")
                        f1_score_ftrl_lr = f1_score(val_y, predictions)
                        logger.info(f1_score_ftrl_lr)
                        logger.info("lR FTRL accuracy_score")
                        accuracy_score_ftrl_lr = accuracy_score(val_y, predictions)
                        logger.info(accuracy_score_ftrl_lr)

                        # training fm_ftrl
                        logger.info("Factorization Machine FTRL Training......")
                        params_ftrl_fm["interaction"] = True
                        secure_ftrl_fm = SecureFTRL(**params_ftrl_fm)
                        score_ftrl_fm = [0.] * len(trn_y)
                        auc_trn_100_ftrl_fm = []
                        for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                            p = secure_ftrl_fm.predict(X)
                            score_ftrl_fm[i] = p
                            secure_ftrl_fm.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                                eval_metric = roc_auc_score(trn_y[:(i + 1)], score_ftrl_fm[:(i + 1)])
                                auc_trn_100_ftrl_fm.append(eval_metric)
                        score_ftrl_fm = [0.] * len(val_y)
                        auc_val_100_ftrl_fm = []
                        for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                            p = secure_ftrl_fm.predict(X)
                            score_ftrl_fm[i] = p
                            secure_ftrl_fm.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                                eval_metric = roc_auc_score(val_y[:(i + 1)], score_ftrl_fm[:(i + 1)])
                                auc_val_100_ftrl_fm.append(eval_metric)
                        predictions = list(map(round, score_ftrl_fm))
                        logger.info("FM FTRL roc_auc_score")
                        roc_auc_score_ftrl_fm = roc_auc_score(val_y, score_ftrl_fm)
                        logger.info(roc_auc_score_ftrl_fm)
                        logger.info("FM FTRL f1_score")
                        f1_score_ftrl_fm = f1_score(val_y, predictions)
                        logger.info(f1_score_ftrl_fm)
                        logger.info("FM FTRL accuracy_score")
                        accuracy_score_ftrl_fm = accuracy_score(val_y, predictions)
                        logger.info(accuracy_score_ftrl_fm)

                        # training reg_fm_ftrl exp_reg
                        logger.info("RegFM FTRL Training......")
                        params_reg_fm["exp_reg"] = True
                        secure_exp_reg_fm = SecureRegFTRL(**params_reg_fm)
                        score_exp_reg_fm = [0.] * len(trn_y)
                        auc_trn_100_exp_reg_fm = []
                        for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                            p = secure_exp_reg_fm.predict(X)
                            score_exp_reg_fm[i] = p
                            secure_exp_reg_fm.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                                eval_metric = roc_auc_score(trn_y[:(i + 1)], score_exp_reg_fm[:(i + 1)])
                                auc_trn_100_exp_reg_fm.append(eval_metric)
                        score_exp_reg_fm = [0.] * len(val_y)
                        auc_val_100_exp_reg_fm = []
                        for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                            p = secure_exp_reg_fm.predict(X)
                            score_exp_reg_fm[i] = p
                            secure_exp_reg_fm.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                                eval_metric = roc_auc_score(val_y[:(i + 1)], score_exp_reg_fm[:(i + 1)])
                                auc_val_100_exp_reg_fm.append(eval_metric)
                        predictions = list(map(round, score_exp_reg_fm))
                        logger.info("ExpRegFM FTRL roc_auc_score")
                        roc_auc_score_exp_reg_fm = roc_auc_score(val_y, score_exp_reg_fm)
                        logger.info(roc_auc_score_exp_reg_fm)
                        logger.info("ExpRegFM FTRL f1_score")
                        f1_score_exp_reg_fm = f1_score(val_y, predictions)
                        logger.info(f1_score_exp_reg_fm)
                        logger.info("ExpRegFM FTRL accuracy_score")
                        accuracy_score_exp_reg_fm = accuracy_score(val_y, predictions)
                        logger.info(accuracy_score_exp_reg_fm)

                        # training reg_fm_ftrl inv_reg
                        params_reg_fm["exp_reg"] = False
                        logger.info("InvRegFM FTRL Training......")
                        secure_inv_reg_fm = SecureRegFTRL(**params_reg_fm)
                        score_inv_reg_fm = [0.] * len(trn_y)
                        auc_trn_100_inv_reg_fm = []
                        for i, (X, y) in enumerate(data_generator(trn_X, trn_y)):
                            p = secure_inv_reg_fm.predict(X)
                            score_inv_reg_fm[i] = p
                            secure_exp_reg_fm.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.75 / 100) == 0:
                                eval_metric = roc_auc_score(trn_y[:(i + 1)], score_inv_reg_fm[:(i + 1)])
                                auc_trn_100_inv_reg_fm.append(eval_metric)
                        score_inv_reg_fm = [0.] * len(val_y)
                        auc_val_100_inv_reg_fm = []
                        for i, (X, y) in enumerate(data_generator(val_X, val_y)):
                            p = secure_inv_reg_fm.predict(X)
                            score_inv_reg_fm[i] = p
                            secure_inv_reg_fm.update(X, p, y)
                            if i > 0 and i % (sample_size * 0.25 / 100) == 0:
                                eval_metric = roc_auc_score(val_y[:(i + 1)], score_inv_reg_fm[:(i + 1)])
                                auc_val_100_inv_reg_fm.append(eval_metric)
                        predictions = list(map(round, score_inv_reg_fm))
                        logger.info("InvRegFM FTRL roc_auc_score")
                        roc_auc_score_inv_reg_fm = roc_auc_score(val_y, score_inv_reg_fm)
                        logger.info(roc_auc_score_inv_reg_fm)
                        logger.info("InvRegFM FTRL f1_score")
                        f1_score_inv_reg_fm = f1_score(val_y, predictions)
                        logger.info(f1_score_inv_reg_fm)
                        logger.info("InvRegFM FTRL accuracy_score")
                        accuracy_score_inv_reg_fm = accuracy_score(val_y, predictions)
                        logger.info(accuracy_score_inv_reg_fm)

                        # logger.info("LightGBM Training......")
                        # lgb_trn = lgb.Dataset(
                        #     data=trn_X,
                        #     label=trn_y,
                        # )
                        # lgb_val = lgb.Dataset(
                        #     data=val_X,
                        #     label=val_y,
                        #     reference=lgb_trn
                        # )
                        # lgb_booster = lgb.train(
                        #     params=params_lgb,
                        #     train_set=lgb_trn,
                        #     num_boost_round=10000,
                        #     valid_sets=[lgb_trn, lgb_val],
                        #     early_stopping_rounds=200,
                        #     verbose_eval=None,
                        #     keep_training_booster=True
                        # )
                        # score_lgb = lgb_booster.predict(val_X)
                        # predictions = list(map(round, score_lgb))
                        # logger.info("lgb roc_auc_score")
                        # lgb_roc_auc_score = roc_auc_score(val_y, score_lgb)
                        # logger.info(lgb_roc_auc_score)
                        # logger.info("lgb f1_score")
                        # lgb_f1_score = f1_score(val_y, predictions)
                        # logger.info(lgb_f1_score)
                        # logger.info("lgb accuracy_score")
                        # lgb_accuracy_score = accuracy_score(val_y, predictions)
                        # logger.info(lgb_accuracy_score)

                        num_cut_valid = data_simulator.k_cut_valid
                        num_w_fst_simulator = len(data_simulator.w_fst_index)
                        num_w_snd_simulator = len(data_simulator.w_snd_index)
                        num_w_snd_related_simulator = len(data_simulator.w_snd_related_index)
                        num_w_snd_valid = len(secure_ftrl_fm.w_fm.values())

                        num_w_fst_ftrl_lr = sum(_ != 0 for _ in secure_ftrl_lr.w)
                        num_w_fst_ftrl_fm = sum(_ != 0 for _ in secure_ftrl_fm.w)
                        num_w_fst_exp_reg_fm = sum(_ != 0 for _ in secure_exp_reg_fm.w)
                        num_w_fst_inv_reg_fm = sum(_ != 0 for _ in secure_inv_reg_fm.w)

                        num_w_fst_ftrl_lr_valid = len(
                            set(data_simulator.w_fst_index) &
                            set([i for i in range(int(params_ftrl_fm["D"])) if secure_ftrl_lr.w[i] != 0])
                        )
                        num_w_fst_ftrl_fm_valid = len(
                            set(data_simulator.w_fst_index) &
                            set([i for i in range(int(params_ftrl_fm["D"])) if secure_ftrl_fm.w[i] != 0])
                        )
                        num_w_fst_exp_reg_fm_valid = len(
                            set(data_simulator.w_fst_index) &
                            set([i for i in range(int(params_ftrl_fm["D"])) if secure_exp_reg_fm.w[i] != 0])
                        )
                        num_w_fst_inv_reg_fm_valid = len(
                            set(data_simulator.w_fst_index) &
                            set([i for i in range(int(params_ftrl_fm["D"])) if secure_inv_reg_fm.w[i] != 0])
                        )

                        w_snd_ftrl_fm = secure_ftrl_fm.w_fm.values()
                        w_snd_exp_reg_fm = secure_exp_reg_fm.w_fm.values()
                        w_snd_inv_reg_fm = secure_inv_reg_fm.w_fm.values()

                        #         w_snd_ftrl_fm_valid = [secure_ftrl_fm.w_fm[index] for index in ]
                        #         w_snd_exp_reg_fm_valid = secure_exp_reg_fm.w_fm.values()
                        #         w_snd_inv_reg_fm_valid = secure_inv_reg_fm.w_fm.values()

                        num_zeros_weight_w_snd_ftrl_fm = sum(sum(_) == 0 for _ in w_snd_ftrl_fm)
                        num_zeros_weight_w_snd_exp_reg_fm = sum(sum(_) == 0 for _ in w_snd_exp_reg_fm)
                        num_zeros_weight_w_snd_inv_reg_fm = sum(sum(_) == 0 for _ in w_snd_inv_reg_fm)

                        num_zeros_w_snd_ftrl_fm = sum(sum(_ == 0 for _ in weight) for weight in w_snd_ftrl_fm)
                        num_zeros_w_snd_exp_reg_fm = sum(sum(_ == 0 for _ in weight) for weight in w_snd_exp_reg_fm)
                        num_zeros_w_snd_inv_reg_fm = sum(sum(_ == 0 for _ in weight) for weight in w_snd_inv_reg_fm)

                        w_snd_ftrl_fm_valid = [item for (key, item) in secure_ftrl_fm.w_fm.items()
                                               if key in data_simulator.w_snd_related_index]
                        w_snd_exp_reg_fm_valid = [item for (key, item) in secure_exp_reg_fm.w_fm.items()
                                                  if key in data_simulator.w_snd_related_index]
                        w_snd_inv_reg_fm_valid = [item for (key, item) in secure_inv_reg_fm.w_fm.items()
                                                  if key in data_simulator.w_snd_related_index]

                        #         w_snd_ftrl_fm_valid = [secure_ftrl_fm.w_fm[index] for index in ]
                        #         w_snd_exp_reg_fm_valid = secure_exp_reg_fm.w_fm.values()
                        #         w_snd_inv_reg_fm_valid = secure_inv_reg_fm.w_fm.values()

                        num_zeros_weight_w_snd_ftrl_fm_valid = sum(sum(_) == 0 for _ in w_snd_ftrl_fm_valid)
                        num_zeros_weight_w_snd_exp_reg_fm_valid = sum(sum(_) == 0 for _ in w_snd_exp_reg_fm_valid)
                        num_zeros_weight_w_snd_inv_reg_fm_valid = sum(sum(_) == 0 for _ in w_snd_inv_reg_fm_valid)

                        num_zeros_w_snd_ftrl_fm_valid = sum(sum(_ == 0 for _ in weight) for weight in w_snd_ftrl_fm_valid)
                        num_zeros_w_snd_exp_reg_fm_valid = sum(sum(_ == 0 for _ in weight) for weight in w_snd_exp_reg_fm_valid)
                        num_zeros_w_snd_inv_reg_fm_valid = sum(sum(_ == 0 for _ in weight) for weight in w_snd_inv_reg_fm_valid)

                        result = dict(zip(columns, [eval(_) for _ in columns]))
                        with open(f"modelResults/{sample_size}/{filename}", "wb") as f:
                            pickle.dump(result, f)
                        pprint(dict(zip(columns_display, [eval(_) for _ in columns_display])))
                        time.sleep(0.5)