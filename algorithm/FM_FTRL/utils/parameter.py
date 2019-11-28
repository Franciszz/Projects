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
    random_state=19960102,
    test_size=0.25,
    feature_fraction_active=0.5,
    num_boost_round_secure=1000,
    num_boost_round_active=5,
    early_stopping_rounds_secure=100
)

params_experiment = {
    "sample_size": int(1e4),
    "D": int(1e7),
    "prob_threshold": 0.90,
    "w_fst_threshold": 0.5,
    "w_snd_threshold": 5,
    "num_valid_feature": 10,
    "num_noisy_feature": 1,
    "num_valid_feature_ratio": 0.6,
    "std_init_weight_fst": 1,
    "std_init_weight_snd": 10,
    "k_cut_tuple_n": (5, 10, 20, 30, 50),
    "k_cut_tuple_p": (0.3, 0.3, 0.2, 0.1, 0.1)
}


params_ftrl_fm_self = {
    "L1": 0.01,
    "L2": 0.1,
    "alpha": 0.01,
    "beta": 0.1,
    "D": int(1e7),
    "L1_fm": 0.1,
    "L2_fm": 0.1,
    "fm_dim": 3,
    "fm_initDev": 0.1,
    "alpha_fm": 0.1,
    "beta_fm": 0.1,
    "interaction": True,
    "dropout_rate": 1.0
}


params_reg_fm_self = {
    "L1": 0.01,
    "L2": 0.1,
    "alpha": 0.01,
    "beta": 0.1,
    "D": int(1e7),
    "L1_fm": 0.1,
    "L2_fm": 0.1,
    "fm_dim": 5,
    "fm_initDev": 0.1,
    "alpha_fm": 0.1,
    "beta_fm": 0.1,
    "interaction": True,
    "dropout_rate": 1.0
}


params_ftrl_fm = {
    "L1": 0.25,
    "L2": 1,
    "alpha": 0.1,
    "beta": 0.1,
    "D": int(1e7),
    "L1_fm": 0.1,
    "L2_fm": 1,
    "fm_dim": 3,
    "fm_initDev": 0.1,
    "alpha_fm": 0.01,
    "beta_fm": 0.1,
    "interaction": True,
    "dropout_rate": 1.0
}


params_reg_fm = {
    "L1": 0.25,
    "L2": 1,
    "alpha": 0.1,
    "beta": 0.1,
    "D": int(1e7),
    "L1_fm": 0.1,
    "L2_fm": 1,
    "fm_dim": 3,
    "exp_reg": True,
    "fm_initDev": 0.1,
    "alpha_fm": 0.015,
    "beta_fm": 0.1,
    "interaction": True,
    "dropout_rate": 1.0
}



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

columns = [
    "sample_size",
    "num_noisy_feature",
    "fm_dim",
    "seed",
    "positive_instance_ratio",

    "auc_trn_100_ftrl_lr",
    "auc_trn_100_ftrl_fm",
    "auc_trn_100_exp_reg_fm",
    "auc_trn_100_inv_reg_fm",

    "auc_val_100_ftrl_lr",
    "auc_val_100_ftrl_fm",
    "auc_val_100_exp_reg_fm",
    "auc_val_100_inv_reg_fm",

    "roc_auc_score_ftrl_lr",
    "roc_auc_score_ftrl_fm",
    "roc_auc_score_exp_reg_fm",
    "roc_auc_score_inv_reg_fm",

    "f1_score_ftrl_lr",
    "f1_score_ftrl_fm",
    "f1_score_exp_reg_fm",
    "f1_score_inv_reg_fm",

    "accuracy_score_ftrl_lr",
    "accuracy_score_ftrl_fm",
    "accuracy_score_exp_reg_fm",
    "accuracy_score_inv_reg_fm",

    # "lgb_roc_auc_score",
    # "lgb_f1_score",
    # "lgb_accuracy_score",

    "num_cut_valid",
    "num_w_fst_simulator",
    "num_w_snd_simulator",
    "num_w_snd_valid",
    "num_w_snd_related_simulator",

    "num_w_fst_ftrl_lr",
    "num_w_fst_ftrl_fm",
    "num_w_fst_exp_reg_fm",
    "num_w_fst_inv_reg_fm",

    "num_w_fst_ftrl_lr_valid",
    "num_w_fst_ftrl_fm_valid",
    "num_w_fst_exp_reg_fm_valid",
    "num_w_fst_inv_reg_fm_valid",

    "num_zeros_weight_w_snd_ftrl_fm",
    "num_zeros_weight_w_snd_exp_reg_fm",
    "num_zeros_weight_w_snd_inv_reg_fm",

    "num_zeros_w_snd_ftrl_fm",
    "num_zeros_w_snd_exp_reg_fm",
    "num_zeros_w_snd_inv_reg_fm",

    "num_zeros_weight_w_snd_ftrl_fm_valid",
    "num_zeros_weight_w_snd_exp_reg_fm_valid",
    "num_zeros_weight_w_snd_inv_reg_fm_valid",

    "num_zeros_w_snd_ftrl_fm_valid",
    "num_zeros_w_snd_exp_reg_fm_valid",
    "num_zeros_w_snd_inv_reg_fm_valid"
]

columns_display = [
    "sample_size",
    "num_noisy_feature",
    "fm_dim",
    "seed",
    "positive_instance_ratio",

    # "auc_trn_100_ftrl_lr",
    # "auc_trn_100_ftrl_fm",
    # "auc_trn_100_exp_reg_fm",
    # "auc_trn_100_inv_reg_fm",
    #
    # "auc_val_100_ftrl_lr",
    # "auc_val_100_ftrl_fm",
    # "auc_val_100_exp_reg_fm",
    # "auc_val_100_inv_reg_fm",

    "roc_auc_score_ftrl_lr",
    "roc_auc_score_ftrl_fm",
    "roc_auc_score_exp_reg_fm",
    "roc_auc_score_inv_reg_fm",

    "f1_score_ftrl_lr",
    "f1_score_ftrl_fm",
    "f1_score_exp_reg_fm",
    "f1_score_inv_reg_fm",

    "accuracy_score_ftrl_lr",
    "accuracy_score_ftrl_fm",
    "accuracy_score_exp_reg_fm",
    "accuracy_score_inv_reg_fm",

    # "lgb_roc_auc_score",
    # "lgb_f1_score",
    # "lgb_accuracy_score",

    "num_cut_valid",
    "num_w_fst_simulator",
    "num_w_snd_simulator",
    "num_w_snd_valid",
    "num_w_snd_related_simulator",

    "num_w_fst_ftrl_lr",
    "num_w_fst_ftrl_fm",
    "num_w_fst_exp_reg_fm",
    "num_w_fst_inv_reg_fm",

    "num_w_fst_ftrl_lr_valid",
    "num_w_fst_ftrl_fm_valid",
    "num_w_fst_exp_reg_fm_valid",
    "num_w_fst_inv_reg_fm_valid",

    "num_zeros_weight_w_snd_ftrl_fm",
    "num_zeros_weight_w_snd_exp_reg_fm",
    "num_zeros_weight_w_snd_inv_reg_fm",

    "num_zeros_w_snd_ftrl_fm",
    "num_zeros_w_snd_exp_reg_fm",
    "num_zeros_w_snd_inv_reg_fm",

    "num_zeros_weight_w_snd_ftrl_fm_valid",
    "num_zeros_weight_w_snd_exp_reg_fm_valid",
    "num_zeros_weight_w_snd_inv_reg_fm_valid",

    "num_zeros_w_snd_ftrl_fm_valid",
    "num_zeros_w_snd_exp_reg_fm_valid",
    "num_zeros_w_snd_inv_reg_fm_valid"
]