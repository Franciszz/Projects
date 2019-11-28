#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: parameter.py
@time: 2019-11-17 13:56
@desc:
"""

# please attach to (https://lightgbm.apachecn.org/#/docs/6)
params_global = dict(
    random_state=19960101,
    test_size=0.25,
    feature_fraction_active=0.5,
    num_boost_round_secure=1000,
    num_boost_round_active=5,
    early_stopping_rounds_secure=100
)


params_lgb = {
    'task': 'train',
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'binary',  # 因为要完成预测用户是否买单行为，所以是binary，不买是0，购买是1
    'metric': 'auc',  # 评判指标
    'max_bin': 255,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.01,  # 学习率
    'num_leaves': 36,  # 大会更准,但可能过拟合
    'max_depth': 8,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.8,  # 防止过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.8,  # 防止过拟合
    'min_data_in_leaf': 21,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'header': True  # 数据集是否带表
}

params_lgb_all = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_bin": 255,
    "num_leaves": 36,
    # "num_iterations": 100,
    "learning_rate": 0.01,

    "max_depth": -1,
    "min_data_in_leaf": 21,
    "min_sum_hessian_in_leaf": 3.0,
    "min_split_gain": 0,

    "feature_fraction": 0.8,
    "feature_fraction_seed": 19960101,
    "bagging_freq": 5,
    "bagging_fraction": 0.8,
    "bagging_seed": 19960101,
    "lambda_l1": 0,
    "lambda_l2": 0,

    "header": True,
    "ignore_column": ""
}