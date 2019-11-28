#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: model.py
@time: 2019-11-17 13:57
@desc:
"""

import os
import lightgbm as lgb


class SecureLightGBM(object):
    """
    This is the framework of SecureLightGBM based on LightGBM algorithm, which can be applied to
    privacy-preserving modeling application.
    """

    def __init__(self, model_dir):
        if not os.path.exists("modelPath/"):
            os.mkdir("modelPath")
        self.model_dir = "modelPath/" + model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def set_train(self, trn_x, val_x, trn_y, val_y, params, feature_fraction_active=0.5,
                  num_boost_round_secure=1000, num_boost_round_active=10,
                  early_stopping_rounds_secure=100):
        num_feature_active = round(feature_fraction_active * trn_x.shape[1])
        lgb_trn_active = lgb.Dataset(
            data=trn_x.iloc[:, :num_feature_active],
            label=trn_y,
            free_raw_data=True
        )
        lgb_val_active = lgb.Dataset(
            data=val_x.iloc[:, :num_feature_active],
            label=val_y,
            reference=lgb_trn_active,
            free_raw_data=True
        )
        lgb_booster_active = lgb.train(
            params=params,
            train_set=lgb_trn_active,
            num_boost_round=num_boost_round_active,
            valid_sets=[lgb_trn_active, lgb_val_active],
            early_stopping_rounds=None,
            verbose_eval=1,
            keep_training_booster=True
        )
        lgb_booster_active.save_model(f"{self.model_dir}/SecureLightGBM_Active")
        lgb_trn_secure = lgb.Dataset(
            data=trn_x,
            label=trn_y
        )
        lgb_val_secure = lgb.Dataset(
            data=val_x,
            label=val_y,
            reference=lgb_trn_secure
        )
        lgb_booster_secure = lgb.train(
            params=params,
            train_set=lgb_trn_secure,
            init_model=lgb_booster_active,
            num_boost_round=num_boost_round_secure,
            valid_sets=[lgb_trn_secure, lgb_val_secure],
            early_stopping_rounds=early_stopping_rounds_secure,
            verbose_eval=int(0.5 * early_stopping_rounds_secure),
            keep_training_booster=True
        )
        lgb_booster_secure.save_model(f"{self.model_dir}/SecureLightGBM_Complete")
        lgb_trn_original = lgb.Dataset(
            data=trn_x,
            label=trn_y
        )
        lgb_val_original = lgb.Dataset(
            data=val_x,
            label=val_y,
            reference=lgb_trn_original
        )
        lgb_booster_original = lgb.train(
            params=params,
            train_set=lgb_trn_original,
            num_boost_round=num_boost_round_secure,
            valid_sets=[lgb_trn_original, lgb_val_original],
            early_stopping_rounds=early_stopping_rounds_secure,
            verbose_eval=int(0.5 * early_stopping_rounds_secure),
            keep_training_booster=True
        )
        lgb_booster_original.save_model(f"{self.model_dir}/SecureLightGBM_Original")
        return lgb_booster_original, lgb_booster_secure, lgb_booster_active