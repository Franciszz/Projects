#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: experiment.py
@time: 2019-11-19 15:09
@desc:
"""

import os
import time
import math

import zipfile
import pandas as pd
pd.options.display.max_columns = 20

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from pprint import pprint

from utils.logger import Logger
from utils.parameter import params_global, params_ftrl
from utils.featurizer import qcut
from utils.model import SecureFTRL, data_generator
from utils.model_backup import SecureLR, data_generator

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
    df = pd.read_csv("data/GiveMeSomeCredit/cs-training.csv").iloc[:, 1:]
    for col in df.columns:
        if df[col].nunique() > 100:
            df[col] = qcut(df[col], int(math.sqrt(df[col].nunique())))
    trn_x, val_x, trn_y, val_y = train_test_split(
        df.iloc[:, 1:],
        df.iloc[:, 0],
        test_size=params_global["test_size"],
        random_state=params_global["random_state"],
        stratify=df["SeriousDlqin2yrs"]
    )
    logger.info("global parameters & SecureFTRL")
    logger.info(params_ftrl)
    secure_fm = SecureFTRL(**params_ftrl)
    scores = []
    for i, (x, y) in enumerate(data_generator(trn_x, trn_y)):
        p = secure_fm.predict(x)
        scores.append(p)
        secure_fm.update(x, p, y)
        if i > 0 and i % 6000 == 0:
            print(i, roc_auc_score(trn_y[:(i+1)], scores))
    scores = []
    for i, (x, y) in enumerate(data_generator(val_x, val_y)):
        p = secure_fm.predict(x)
        scores.append(p)
        if i > 0 and i % 2000 == 0:
            print(i, roc_auc_score(val_y[:(i+1)], scores))
    predictions = list(map(round, scores))
    logger.info("roc_auc_score\n")
    logger.info(roc_auc_score(val_y, scores))
    logger.info("f1_score")
    logger.info(f1_score(val_y, predictions))
    logger.info("accuracy_score")
    logger.info(accuracy_score(val_y, predictions))
    # pprint(secure_fm.w_fm)


