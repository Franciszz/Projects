#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: experiment.py
@time: 2019-11-17 13:57
@desc:
"""

import os
import time

import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, auc, roc_auc_score, log_loss, accuracy_score
from pprint import pprint

from utils.logger import Logger
from utils.parameters import params_global, params_lgb
from utils.model import SecureLightGBM


# set logger
marks = time.strftime("%Y-%m-%d %H:%M")
logger = Logger(filename="GiveMeSomeCredit", dir_name=f"logs/{marks}").set_logger()


# this is the procedure using the GiveMeSomeCredit dataset as example
def unzip_file(src_file, dest_dir, pwd=False):
    src = zipfile.ZipFile(src_file)
    try:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        src.extractall(path=dest_dir, pwd=pwd)
    except RuntimeError as e:
        logger.infor(e)
    src.close()


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/GiveMeSomeCredit.zip"):
        os.system("kaggle competitions download -c GiveMeSomeCredit -p data")
        unzip_file("data/GiveMeSomeCredit.zip", "data/GiveMeSomeCredit")
    df = pd.read_csv("data/GiveMeSomeCredit/cs-training.csv").iloc[:, 1:].fillna(0)
    trn_x, val_x, trn_y, val_y = train_test_split(
        df.iloc[:, 1:],
        df.iloc[:, 0],
        test_size=params_global["test_size"],
        random_state=params_global["random_state"],
        stratify=df["SeriousDlqin2yrs"]
    )
    secure_lgb = SecureLightGBM(time.strftime("%Y-%m-%d %H:%M", time.localtime()))
    lgb_original, lgb_secure, lgb_active = secure_lgb.set_train(
        trn_x=trn_x,
        val_x=val_x,
        trn_y=trn_y,
        val_y=val_y,
        params=params_lgb,
        feature_fraction_active=params_global["feature_fraction_active"],
        num_boost_round_secure=params_global["num_boost_round_secure"],
        num_boost_round_active=params_global["num_boost_round_active"],
        early_stopping_rounds_secure=params_global["early_stopping_rounds_secure"]
    )
    lr_boost = LogisticRegression(penalty='l2', fit_intercept=True)
    lr_boost.fit(trn_x, trn_y)
    logger.info("Global Parameters\n")
    logger.info(params_global)
    logger.info("LightGBM Parameters\n")
    logger.info(params_lgb)
    logger.info("Header of lgb models predictions\n")
    predictions = pd.DataFrame(
            {
                "val_y":
                    val_y,
                "val_cls_original":
                    lgb_original.predict(val_x),
                "val_cls_secure":
                    lgb_secure.predict(val_x),
                "val_cls_active":
                    lgb_active.predict(val_x.iloc[:, :int(params_global["feature_fraction_active"] * val_x.shape[1])]),
                "val_lr":
                    lr_boost.predict_proba(val_x)[:, 1]
            }
        ).sort_index()
    logger.info(predictions.head(12))
    logger.info("Metrics of lgb models predictions\n")
    metrics = pd.DataFrame(
        {
            "f1_score":
                [f1_score(predictions["val_y"], predictions[pred].round()) for pred in predictions.columns[1:]],
            "accuracy":
                [accuracy_score(predictions["val_y"], predictions[pred].round()) for pred in predictions.columns[1:]],
            "roc_auc_score":
                [roc_auc_score(predictions["val_y"], predictions[pred]) for pred in predictions.columns[1:]]
        },
        index=["original", "secure", "active", "lr"]
    )
    logger.info(metrics)




