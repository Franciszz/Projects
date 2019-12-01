#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: credit_transform.py
@time: 2019-11-30 18:46
@desc:
"""

import os
import time
import math
import gc

import pickle
import zipfile
import numpy as np
import pandas as pd
pd.options.display.max_columns = 20

from steppy.base import BaseTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from pprint import pprint

from utils.logger import Logger
from utils.parameter import params_global, params_ftrl_fm, params_reg_fm
from utils.featurizer import qcut
from utils.model import SecureFTRL, data_generator


# set logger
marks = time.strftime("%Y-%m-%d %H:%M")
logger = Logger(filename="GiveMeSomeCredit", dir_name=f"logs/{marks}").set_logger()


def unzip_file(src_file, dest_dir, pwd=False):
    src = zipfile.ZipFile(src_file)
    try:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        src.extractall(path=dest_dir, pwd=pwd)
    except RuntimeError as e:
        logger.infor(e)
    src.close()


def ApplicationFeatureExtract(data):
    df = data.loc[:, ["SK_ID_CURR", "TARGET"]]
    # 合约类型
    df['NAME_CONTRACT_TYPE'] = data['NAME_CONTRACT_TYPE']. \
        astype('category').cat.rename_categories([0, 1])
    # 性别
    df['CODE_GENDER'] = data['CODE_GENDER']. \
        replace('XNA', np.nan). \
        astype('category').cat.rename_categories([0, 1])
    # 是否有车
    df['FLAG_OWN_CAR'] = data['FLAG_OWN_CAR']. \
        astype('category').cat.rename_categories([0, 1])
    # 是否有房地产
    df['FLAG_OWN_REALTY'] = data['FLAG_OWN_REALTY']. \
        astype('category').cat.rename_categories([0, 1])
    #
    df['EMERGENCYSTATE_MODE'] = data['EMERGENCYSTATE_MODE']. \
        astype('category').cat.rename_categories([0, 1])
    # 申请工作日
    df['WEEKDAY_APPR_PROCESS_START'] = data['WEEKDAY_APPR_PROCESS_START']. \
        astype('category').cat.rename_categories(range(7))

    df['FONDKAPREMONT_MODE'] = data['FONDKAPREMONT_MODE']. \
        astype('category').cat.rename_categories(range(4))

    df['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE']. \
        astype('category').cat.rename_categories(range(7))
    # 收入类型
    df['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE']. \
        astype('category').cat.rename_categories(range(8))
    # 教育情况
    df['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE']. \
        astype('category').cat.rename_categories(range(5))
    # 婚姻状况
    df['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS']. \
        replace('Unknown', np.nan). \
        astype('category').cat.rename_categories(range(5))
    # 住房类型
    df['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE']. \
        astype('category').cat.rename_categories(range(6))
    df['HOUSETYPE_MODE'] = data['HOUSETYPE_MODE']. \
        astype('category').cat.rename_categories(range(3))
    df['WALLSMATERIAL_MODE'] = data['WALLSMATERIAL_MODE']. \
        astype('category').cat.rename_categories(range(7))
    # 职业类型
    df['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE']. \
        astype('category').cat.rename_categories(range(18))
    # 机构类型
    df['ORGANIZATION_TYPE'] = data['ORGANIZATION_TYPE']. \
        replace('XNA', np.nan). \
        astype('category').cat.rename_categories(range(57))

    data = data.replace(365243, np.nan).rename(columns={
        'CODE_GENDER': 'BASE_CODE_GENDER',
        'CNT_CHILDREN': 'BASE_CNT_CHILDREN',
        'CNT_FAM_MEMBERS': 'BASE_CNT_FAM_MEMBERS',
        'OCCUPATION_TYPE': 'BASE_OCCUPATION_TYPE',
        'ORGANIZATION_TYPE': 'BASE_ORGANIZATION_TYPE',
        'DEF_30_CNT_SOCIAL_CIRCLE': 'BASE_DEF_30_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE': 'BASE_DEF_60_CNT_SOCIAL_CIRCLE',
        'OBS_30_CNT_SOCIAL_CIRCLE': 'BASE_OBS_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE': 'BASE_OBS_60_CNT_SOCIAL_CIRCLE',

        'REGION_POPULATION_RELATIVE': 'EXT_REGION_POPULATION_RELATIVE',
        'REGION_RATING_CLIENT': 'EXT_REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY': 'EXT_REGION_RATING_CLIENT_W_CITY',

        'OWN_CAR_AGE': 'DAYS_OWN_CAR_AGE',

        'LIVE_CITY_NOT_WORK_CITY': 'FLAG_LIVE_CITY_NOT_WORK_CITY',
        'LIVE_REGION_NOT_WORK_REGION': 'FLAG_LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY': 'FLAG_REG_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_WORK_REGION': 'FLAG_REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY': 'FLAG_REG_CITY_NOT_LIVE_CITY',
        'REG_REGION_NOT_LIVE_REGION': 'FLAG_REG_REGION_NOT_LIVE_REGION',

        'HOUR_APPR_PROCESS_START': 'NAME_HOUR_APPR_PROCESS_START',
        'WEEKDAY_APPR_PROCESS_START': 'NAME_WEEKDAY_APPR_PROCESS_START',

        'APARTMENTS_MODE': 'ASSET_APARTMENTS_MODE',
        'BASEMENTAREA_MODE': 'ASSET_BASEMENTAREA_MODE',
        'COMMONAREA_MODE': 'ASSET_COMMONAREA_MODE',
        'ELEVATORS_MODE': 'ASSET_ELEVATORS_MODE',
        'ENTRANCES_MODE': 'ASSET_ENTRANCES_MODE',
        'EMERGENCYSTATE_MODE': 'ASSET_EMERGENCYSTATE_MODE',
        'FLOORSMAX_MODE': 'ASSET_FLOORSMAX_MODE',
        'FLOORSMIN_MODE': 'ASSET_FLOORSMIN_MODE',
        'FONDKAPREMONT_MODE': 'ASSET_FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE': 'ASSET_HOUSETYPE_MODE',
        'LANDAREA_MODE': 'ASSET_LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE': 'ASSET_LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE': 'ASSET_LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE': 'ASSET_NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE': 'ASSET_NONLIVINGAREA_MODE',
        'TOTALAREA_MODE': 'ASSET_TOTALAREA_MODE',
        'WALLSMATERIAL_MODE': 'ASSET_WALLSMATERIAL_MODE',
        'YEARS_BUILD_MODE': 'ASSET_YEARS_BUILD_MODE',
        'YEARS_BEGINEXPLUATATION_MODE': 'ASSET_YEARS_BEGINEXPLUATATION_MODE',

        'APARTMENTS_AVG': 'ASSET_APARTMENTS_AVG',
        'BASEMENTAREA_AVG': 'ASSET_BASEMENTAREA_AVG',
        'COMMONAREA_AVG': 'ASSET_COMMONAREA_AVG',
        'ELEVATORS_AVG': 'ASSET_ELEVATORS_AVG',
        'ENTRANCES_AVG': 'ASSET_ENTRANCES_AVG',
        'FLOORSMAX_AVG': 'ASSET_FLOORSMAX_AVG',
        'FLOORSMIN_AVG': 'ASSET_FLOORSMIN_AVG',
        'LANDAREA_AVG': 'ASSET_LANDAREA_AVG',
        'LIVINGAPARTMENTS_AVG': 'ASSET_LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG': 'ASSET_LIVINGAREA_AVG',
        'NONLIVINGAPARTMENTS_AVG': 'ASSET_NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG': 'ASSET_NONLIVINGAREA_AVG',
        'YEARS_BUILD_AVG': 'ASSET_YEARS_BUILD_AVG',
        'YEARS_BEGINEXPLUATATION_AVG': 'ASSET_YEARS_BEGINEXPLUATATION_AVG',

        'APARTMENTS_MEDI': 'ASSET_APARTMENTS_MEDI',
        'BASEMENTAREA_MEDI': 'ASSET_BASEMENTAREA_MEDI',
        'COMMONAREA_MEDI': 'ASSET_COMMONAREA_MEDI',
        'ELEVATORS_MEDI': 'ASSET_ELEVATORS_MEDI',
        'EMERGENCYSTATE_MEDI': 'ASSET_EMERGENCYSTATE_MEDI',
        'ENTRANCES_MEDI': 'ASSET_ENTRANCES_MEDI',
        'FLOORSMAX_MEDI': 'ASSET_FLOORSMAX_MEDI',
        'FLOORSMIN_MEDI': 'ASSET_FLOORSMIN_MEDI',
        'LANDAREA_MEDI': 'ASSET_LANDAREA_MEDI',
        'LIVINGAPARTMENTS_MEDI': 'ASSET_LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI': 'ASSET_LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI': 'ASSET_NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI': 'ASSET_NONLIVINGAREA_MEDI',
        'YEARS_BUILD_MEDI': 'ASSET_YEARS_BUILD_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI': 'ASSET_YEARS_BEGINEXPLUATATION_MEDI'
    })
    var_docu = [
        'FLAG_DOCUMENT_%d' % i for i in range(2, 22)]
    var_tele = [
        'FLAG_%s' % name for name in
        ('CONT_MOBILE', 'EMAIL', 'MOBIL', 'PHONE', 'WORK_PHONE')
    ]
    var_addr = [
        'FLAG_LIVE_CITY_NOT_WORK_CITY',
        'FLAG_LIVE_REGION_NOT_WORK_REGION',
        'FLAG_REG_CITY_NOT_WORK_CITY',
        'FLAG_REG_REGION_NOT_WORK_REGION',
        'FLAG_REG_CITY_NOT_LIVE_CITY',
        'FLAG_REG_REGION_NOT_LIVE_REGION'
    ]
    var_date = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    df['BASE_child_income_ratio'] = \
        data['BASE_CNT_CHILDREN'] / data['AMT_INCOME_TOTAL']
    df['BASE_income_members_ratio'] = \
        data['BASE_CNT_FAM_MEMBERS'] / data['AMT_INCOME_TOTAL']
    df['BASE_child_credit_ratio'] = \
        data['BASE_CNT_CHILDREN'] / data['AMT_CREDIT']
    df['BASE_members_credit_ratio'] = \
        data['BASE_CNT_FAM_MEMBERS'] / data['AMT_CREDIT']
    df['BASE_members_area_ratio'] = \
        data['BASE_CNT_FAM_MEMBERS'] / data['ASSET_LIVINGAREA_MODE']

    df['AMT_CREDIT'] = data['AMT_CREDIT']
    df['AMT_INCOME_TOTAL'] = data['AMT_INCOME_TOTAL']
    df['AMT_GOODS_PRICE'] = data['AMT_GOODS_PRICE']

    df['AMT_annuity_income_ratio'] = \
        data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    df['AMT_annuity_credit_ratio'] = \
        data['AMT_ANNUITY'] / data['AMT_CREDIT']
    df['AMT_credit_price_ratio'] = \
        data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
    df['AMT_credit_income_ratio'] = \
        data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    df['AMT_price_income_ratio'] = \
        data['AMT_GOODS_PRICE'] / data['AMT_INCOME_TOTAL']

    df["DAYS_BIRTH"] = data['DAYS_BIRTH']
    df["DAYS_EMPLOYED"] = data['DAYS_EMPLOYED']
    df["DAYS_OWN_CAR_AGE"] = data['DAYS_OWN_CAR_AGE']

    df['DAYS_employed_birth_ratio'] = \
        data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    df['DAYS_car_birth_ratio'] = \
        data['DAYS_OWN_CAR_AGE'] / data['DAYS_BIRTH']
    df['DAYS_car_employed_ratio'] = \
        data['DAYS_OWN_CAR_AGE'] / data['DAYS_EMPLOYED']

    df['EXT_SOURCE_MAX'] = \
        data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['EXT_SOURCE_MIN'] = \
        data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['EXT_SOURCE_AVG'] = \
        data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    return df


def ApplicationFeatureExtract(data):
    df = data.loc[:, ["SK_ID_CURR", "TARGET"]]
    # 合约类型
    df['NAME_CONTRACT_TYPE'] = data['NAME_CONTRACT_TYPE']. \
        astype('category').cat.rename_categories([0, 1])
    # 性别
    df['CODE_GENDER'] = data['CODE_GENDER']. \
        replace('XNA', np.nan). \
        astype('category').cat.rename_categories([0, 1])
    # 是否有车
    df['FLAG_OWN_CAR'] = data['FLAG_OWN_CAR']. \
        astype('category').cat.rename_categories([0, 1])
    # 是否有房地产
    df['FLAG_OWN_REALTY'] = data['FLAG_OWN_REALTY']. \
        astype('category').cat.rename_categories([0, 1])
    #
    df['EMERGENCYSTATE_MODE'] = data['EMERGENCYSTATE_MODE']. \
        astype('category').cat.rename_categories([0, 1])
    # 申请工作日
    df['WEEKDAY_APPR_PROCESS_START'] = data['WEEKDAY_APPR_PROCESS_START']. \
        astype('category').cat.rename_categories(range(7))

    df['FONDKAPREMONT_MODE'] = data['FONDKAPREMONT_MODE']. \
        astype('category').cat.rename_categories(range(4))

    df['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE']. \
        astype('category').cat.rename_categories(range(7))
    # 收入类型
    df['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE']. \
        astype('category').cat.rename_categories(range(8))
    # 教育情况
    df['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE']. \
        astype('category').cat.rename_categories(range(5))
    # 婚姻状况
    df['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS']. \
        replace('Unknown', np.nan). \
        astype('category').cat.rename_categories(range(5))
    # 住房类型
    df['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE']. \
        astype('category').cat.rename_categories(range(6))
    df['HOUSETYPE_MODE'] = data['HOUSETYPE_MODE']. \
        astype('category').cat.rename_categories(range(3))
    df['WALLSMATERIAL_MODE'] = data['WALLSMATERIAL_MODE']. \
        astype('category').cat.rename_categories(range(7))
    # 职业类型
    df['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE']. \
        astype('category').cat.rename_categories(range(18))
    # 机构类型
    df['ORGANIZATION_TYPE'] = data['ORGANIZATION_TYPE']. \
        replace('XNA', np.nan). \
        astype('category').cat.rename_categories(range(57))

    data = data.replace(365243, np.nan).rename(columns={
        'CODE_GENDER': 'BASE_CODE_GENDER',
        'CNT_CHILDREN': 'BASE_CNT_CHILDREN',
        'CNT_FAM_MEMBERS': 'BASE_CNT_FAM_MEMBERS',
        'OCCUPATION_TYPE': 'BASE_OCCUPATION_TYPE',
        'ORGANIZATION_TYPE': 'BASE_ORGANIZATION_TYPE',
        'DEF_30_CNT_SOCIAL_CIRCLE': 'BASE_DEF_30_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE': 'BASE_DEF_60_CNT_SOCIAL_CIRCLE',
        'OBS_30_CNT_SOCIAL_CIRCLE': 'BASE_OBS_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE': 'BASE_OBS_60_CNT_SOCIAL_CIRCLE',

        'REGION_POPULATION_RELATIVE': 'EXT_REGION_POPULATION_RELATIVE',
        'REGION_RATING_CLIENT': 'EXT_REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY': 'EXT_REGION_RATING_CLIENT_W_CITY',

        'OWN_CAR_AGE': 'DAYS_OWN_CAR_AGE',

        'LIVE_CITY_NOT_WORK_CITY': 'FLAG_LIVE_CITY_NOT_WORK_CITY',
        'LIVE_REGION_NOT_WORK_REGION': 'FLAG_LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY': 'FLAG_REG_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_WORK_REGION': 'FLAG_REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY': 'FLAG_REG_CITY_NOT_LIVE_CITY',
        'REG_REGION_NOT_LIVE_REGION': 'FLAG_REG_REGION_NOT_LIVE_REGION',

        'HOUR_APPR_PROCESS_START': 'NAME_HOUR_APPR_PROCESS_START',
        'WEEKDAY_APPR_PROCESS_START': 'NAME_WEEKDAY_APPR_PROCESS_START',

        'APARTMENTS_MODE': 'ASSET_APARTMENTS_MODE',
        'BASEMENTAREA_MODE': 'ASSET_BASEMENTAREA_MODE',
        'COMMONAREA_MODE': 'ASSET_COMMONAREA_MODE',
        'ELEVATORS_MODE': 'ASSET_ELEVATORS_MODE',
        'ENTRANCES_MODE': 'ASSET_ENTRANCES_MODE',
        'EMERGENCYSTATE_MODE': 'ASSET_EMERGENCYSTATE_MODE',
        'FLOORSMAX_MODE': 'ASSET_FLOORSMAX_MODE',
        'FLOORSMIN_MODE': 'ASSET_FLOORSMIN_MODE',
        'FONDKAPREMONT_MODE': 'ASSET_FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE': 'ASSET_HOUSETYPE_MODE',
        'LANDAREA_MODE': 'ASSET_LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE': 'ASSET_LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE': 'ASSET_LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE': 'ASSET_NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE': 'ASSET_NONLIVINGAREA_MODE',
        'TOTALAREA_MODE': 'ASSET_TOTALAREA_MODE',
        'WALLSMATERIAL_MODE': 'ASSET_WALLSMATERIAL_MODE',
        'YEARS_BUILD_MODE': 'ASSET_YEARS_BUILD_MODE',
        'YEARS_BEGINEXPLUATATION_MODE': 'ASSET_YEARS_BEGINEXPLUATATION_MODE',

        'APARTMENTS_AVG': 'ASSET_APARTMENTS_AVG',
        'BASEMENTAREA_AVG': 'ASSET_BASEMENTAREA_AVG',
        'COMMONAREA_AVG': 'ASSET_COMMONAREA_AVG',
        'ELEVATORS_AVG': 'ASSET_ELEVATORS_AVG',
        'ENTRANCES_AVG': 'ASSET_ENTRANCES_AVG',
        'FLOORSMAX_AVG': 'ASSET_FLOORSMAX_AVG',
        'FLOORSMIN_AVG': 'ASSET_FLOORSMIN_AVG',
        'LANDAREA_AVG': 'ASSET_LANDAREA_AVG',
        'LIVINGAPARTMENTS_AVG': 'ASSET_LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG': 'ASSET_LIVINGAREA_AVG',
        'NONLIVINGAPARTMENTS_AVG': 'ASSET_NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG': 'ASSET_NONLIVINGAREA_AVG',
        'YEARS_BUILD_AVG': 'ASSET_YEARS_BUILD_AVG',
        'YEARS_BEGINEXPLUATATION_AVG': 'ASSET_YEARS_BEGINEXPLUATATION_AVG',

        'APARTMENTS_MEDI': 'ASSET_APARTMENTS_MEDI',
        'BASEMENTAREA_MEDI': 'ASSET_BASEMENTAREA_MEDI',
        'COMMONAREA_MEDI': 'ASSET_COMMONAREA_MEDI',
        'ELEVATORS_MEDI': 'ASSET_ELEVATORS_MEDI',
        'EMERGENCYSTATE_MEDI': 'ASSET_EMERGENCYSTATE_MEDI',
        'ENTRANCES_MEDI': 'ASSET_ENTRANCES_MEDI',
        'FLOORSMAX_MEDI': 'ASSET_FLOORSMAX_MEDI',
        'FLOORSMIN_MEDI': 'ASSET_FLOORSMIN_MEDI',
        'LANDAREA_MEDI': 'ASSET_LANDAREA_MEDI',
        'LIVINGAPARTMENTS_MEDI': 'ASSET_LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI': 'ASSET_LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI': 'ASSET_NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI': 'ASSET_NONLIVINGAREA_MEDI',
        'YEARS_BUILD_MEDI': 'ASSET_YEARS_BUILD_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI': 'ASSET_YEARS_BEGINEXPLUATATION_MEDI'
    })
    var_docu = [
        'FLAG_DOCUMENT_%d' % i for i in range(2, 22)]
    var_tele = [
        'FLAG_%s' % name for name in
        ('CONT_MOBILE', 'EMAIL', 'MOBIL', 'PHONE', 'WORK_PHONE')
    ]
    var_addr = [
        'FLAG_LIVE_CITY_NOT_WORK_CITY',
        'FLAG_LIVE_REGION_NOT_WORK_REGION',
        'FLAG_REG_CITY_NOT_WORK_CITY',
        'FLAG_REG_REGION_NOT_WORK_REGION',
        'FLAG_REG_CITY_NOT_LIVE_CITY',
        'FLAG_REG_REGION_NOT_LIVE_REGION'
    ]
    var_date = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    df['BASE_child_income_ratio'] = \
        data['BASE_CNT_CHILDREN'] / data['AMT_INCOME_TOTAL']
    df['BASE_income_members_ratio'] = \
        data['BASE_CNT_FAM_MEMBERS'] / data['AMT_INCOME_TOTAL']
    df['BASE_child_credit_ratio'] = \
        data['BASE_CNT_CHILDREN'] / data['AMT_CREDIT']
    df['BASE_members_credit_ratio'] = \
        data['BASE_CNT_FAM_MEMBERS'] / data['AMT_CREDIT']
    df['BASE_members_area_ratio'] = \
        data['BASE_CNT_FAM_MEMBERS'] / data['ASSET_LIVINGAREA_MODE']

    df['AMT_CREDIT'] = data['AMT_CREDIT']
    df['AMT_INCOME_TOTAL'] = data['AMT_INCOME_TOTAL']
    df['AMT_GOODS_PRICE'] = data['AMT_GOODS_PRICE']

    df['AMT_annuity_income_ratio'] = \
        data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    df['AMT_annuity_credit_ratio'] = \
        data['AMT_ANNUITY'] / data['AMT_CREDIT']
    df['AMT_credit_price_ratio'] = \
        data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
    df['AMT_credit_income_ratio'] = \
        data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    df['AMT_price_income_ratio'] = \
        data['AMT_GOODS_PRICE'] / data['AMT_INCOME_TOTAL']

    df["DAYS_BIRTH"] = data['DAYS_BIRTH']
    df["DAYS_EMPLOYED"] = data['DAYS_EMPLOYED']
    df["DAYS_OWN_CAR_AGE"] = data['DAYS_OWN_CAR_AGE']

    df['DAYS_employed_birth_ratio'] = \
        data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    df['DAYS_car_birth_ratio'] = \
        data['DAYS_OWN_CAR_AGE'] / data['DAYS_BIRTH']
    df['DAYS_car_employed_ratio'] = \
        data['DAYS_OWN_CAR_AGE'] / data['DAYS_EMPLOYED']

    df['EXT_SOURCE_MAX'] = \
        data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['EXT_SOURCE_MIN'] = \
        data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['EXT_SOURCE_AVG'] = \
        data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    print(df.shape)
    return df


def PreviousTransform(data):
    df = data.replace({'XNA': np.nan, 365243: np.nan}). \
        sort_values(['SK_ID_CURR', 'SK_ID_PREV'])
    # 消费贷款
    df['NAME_CONTRACT_TYPE_IS_CONSUMER'] = \
        (df['NAME_CONTRACT_TYPE'] == 'Consumer loans').astype('int8')
    # 现金贷款
    df['NAME_CONTRACT_TYPE_IS_CASH'] = \
        (df['NAME_CONTRACT_TYPE'] == 'Cash loans').astype('int8')
    # 租赁贷款
    df['NAME_CONTRACT_TYPE_IS_REVOLVING'] = \
        (df['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int8')

    df['FLAG_LAST_APPL_PER_CONTRACT'] = \
        (df['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y').astype('int8')
    # 是否紧急
    df['NAME_CASH_LOAN_PURPOSE_IS_URGENT'] = \
        (df['NAME_CASH_LOAN_PURPOSE']. \
         isin(['Payments on other loans', 'Urgent needs',
               'Refusal to name the goal'])).astype('int8')
    # 日常消费
    df['NAME_CASH_LOAN_PURPOSE_IS_REGULAR'] = \
        (df['NAME_CASH_LOAN_PURPOSE']. \
         isin(['Everyday expenses', 'Hobby', 'Gasification / water supply',
               'Repairs', 'Car Repairs', 'Journey', 'Medichine',
               'Education', 'Gasification / water supply'])).astype('int8')
    # 商业贷款
    df['NAME_CASH_LOAN_PURPOSE_IS_BUSINESS'] = \
        (df['NAME_CASH_LOAN_PURPOSE']. \
         isin(['Business development',
               'Money for a third person'])).astype('int8')

    # 取消申请
    df['NAME_CONTRACT_IS_CANCEL'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Cancelled').astype('int8')
    # 申请拒绝
    df['NAME_CONTRACT_IS_REFUSED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int8')
    # 贷款未使用
    df['NAME_CONTRACT_IS_UNUSED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Unused offer').astype('int8')
    # 申请通过
    df['NAME_CONTRACT_IS_APPROVED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Approved').astype('int8')

    # 现金还款
    df['NAME_PAYMENT_TYPE_FROM_CASH'] = \
        (df['NAME_PAYMENT_TYPE'] == 'Cash through the bank'). \
            astype('int8')
    # 银行账户还款
    df['NAME_PAYAMENT_TYPE_FROM_ACCOUNT'] = \
        (df['NAME_PAYMENT_TYPE'] == 'Non-cash from your account'). \
            astype('int8')
    # 所在公司还款
    df['NAME_PAYAMENT_TYPE_FROM_EMPLOYER'] = \
        (df['NAME_PAYMENT_TYPE'] == 'Cashless from the account of the employer'). \
            astype('int8')

    df['AMT_CREDIT_LT_APPLICATION'] = \
        (df['AMT_CREDIT'] < df['AMT_APPLICATION']).astype('int8')
    df['DAYS_LAST_DUE_DUE'] = \
        (df['DAYS_LAST_DUE_1ST_VERSION'] < df['DAYS_LAST_DUE']).astype('int8')
    df['DAYS_LAST_DUE_TERMINATION'] = \
        (df['DAYS_LAST_DUE'] < df['DAYS_TERMINATION']).astype('int8')
    df['RATE_HAS_PRIVILEGED'] = \
        (df['RATE_INTEREST_PRIVILEGED'].notna()).astype('int8')
    # num_feature
    df['AMT_CREDIT_APPLICATION_RATIO'] = \
        df['AMT_CREDIT'] / df['AMT_APPLICATION']
    df['AMT_RATE_OF_CREDIT'] = \
        df['AMT_ANNUITY'] * df['CNT_PAYMENT'] / df['AMT_CREDIT']
    return df


def PreviousFeatureExtract(data):
    ori_feature = set(data.columns)
    data = PreviousTransform(data)
    ext_feature = set(data.columns) - ori_feature
    df_groupby = data.groupby(["SK_ID_CURR"])
    df = df_groupby.size().reset_index().rename(columns={0: "Nums_Prev_Application"})
    vc_feature = [
        'NAME_CONTRACT_TYPE',
        'NAME_CASH_LOAN_PURPOSE',
        'NAME_CONTRACT_STATUS',
        'NAME_PAYMENT_TYPE',
        'NAME_TYPE_SUITE',
        'NAME_CLIENT_TYPE',
        'NAME_GOODS_CATEGORY',
        'NAME_PORTFOLIO',
        'NAME_PRODUCT_TYPE',
        'CHANNEL_TYPE',
        'SELLERPLACE_AREA',
        'NAME_SELLER_INDUSTRY',
        'CNT_PAYMENT',
        'NAME_YIELD_GROUP',
        'PRODUCT_COMBINATION'
    ]
    for _f in vc_feature:
        df[f"PREV_{_f}_Nunique"] = df_groupby[_f].nunique()
    for _f in ext_feature:
        df[f"PREV_{_f}_Sum"] = df_groupby[_f].sum()
    num_feature = {
        'AMT_ANNUITY': ["mean"],
        'AMT_APPLICATION': ["max", "mean"],
        'AMT_CREDIT': ["max", "mean"],
        'AMT_DOWN_PAYMENT': ["max", "mean"],
        'AMT_GOODS_PRICE': ["max", "mean"],
        'RATE_DOWN_PAYMENT': ["mean"],
        'DAYS_DECISION': ["max", "mean"],
        'CNT_PAYMENT': ["max", "mean"],
        'DAYS_FIRST_DUE': ["min"],
        'DAYS_LAST_DUE': ["max"],
        'DAYS_TERMINATION': ["mean"],
        'AMT_CREDIT_APPLICATION_RATIO': ["max", "mean"],
        'AMT_RATE_OF_CREDIT': ["mean"]
    }
    for _f, _aggs in num_feature.items():
        for _agg in _aggs:
            df[f"PREV_{_f}_{_agg}"] = df_groupby[_f].agg(_agg)
    print(df.shape)
    return df


def BureauTransform(data):
    df = data.copy()
    df.loc[df['DAYS_CREDIT_ENDDATE'] < -40000,
           'DAYS_CREDIT_ENDDATE'] = 0
    df.loc[df['DAYS_CREDIT_UPDATE'] < -40000,
           'DAYS_CREDIT_UPDATE'] = 0
    df.loc[df['DAYS_ENDDATE_FACT'] < -40000,
           'DAYS_ENDDATE_FACT'] = 0
    df['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    df['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
    df['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
    df['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)
    # cat_feature
    df['CREDIT_ACTIVE_IS_ACTIVE'] = \
        (df['CREDIT_ACTIVE'] == 'Active').astype('int8')
    df['CREDIT_ACTIVE_IS_BAD'] = \
        (df['CREDIT_ACTIVE'] == 'Bad debt').astype('int8')
    df['CREDIT_ACTIVE_IS_CLOSED'] = \
        (df['CREDIT_ACTIVE'] == 'Closed').astype('int8')
    df['CREDIT_ACTIVE_IS_SOLD'] = \
        (df['CREDIT_ACTIVE'] == 'Sold').astype('int8')

    df['CREDIT_TYPE_IS_REVOLVING'] = \
        (df['CREDIT_TYPE'] == 'Credit card').astype('int8')

    # num_feature
    df['AMT_RATIO_ANNUITY_LIMIT'] = \
        df['AMT_ANNUITY'] / df['AMT_CREDIT_SUM_LIMIT']
    df['AMT_RATIO_ANNUITY_SUM'] = \
        df['AMT_ANNUITY'] / df['AMT_CREDIT_SUM']
    df['AMT_RATIO_ANNUITY_DEBT'] = \
        df['AMT_ANNUITY'] / df['AMT_CREDIT_SUM_DEBT']
    df['AMT_RATIO_DEBT_SUM'] = \
        df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM']
    df['AMT_RATIO_LIMIT_SUM'] = \
        df['AMT_CREDIT_SUM_LIMIT'] / df['AMT_CREDIT_SUM']
    df['AMT_RATIO_OVERDUE_SUM'] = \
        df['AMT_CREDIT_SUM_OVERDUE'] / df['AMT_CREDIT_SUM']
    df['AMT_RATIO_MAXOVERDUE_SUM'] = \
        df['AMT_CREDIT_MAX_OVERDUE'] / df['AMT_CREDIT_SUM']
    df['AMT_RATIO_LIMIT_OVERDUE'] = \
        df['AMT_CREDIT_SUM_OVERDUE'] / df['AMT_CREDIT_SUM_LIMIT']
    df['AMT_RATIO_DEBT_OVERDUE'] = \
        df['AMT_CREDIT_SUM_OVERDUE'] / df['AMT_CREDIT_SUM_DEBT']

    df['DAYS_RATIO_OVERDUE_CREDIT'] = \
        df['CREDIT_DAY_OVERDUE'] / df['DAYS_CREDIT']
    #     df['DAYS_RATIO_OVERDUE_FACT'] = \
    #         df['CREDIT_DAY_OVERDUE'] / df['DAYS_CREDIT_ENDDATE']
    return df


def BuearuFeatureExtract(data, balance):
    balance['BUREAU_IS_DPD'] = \
        (balance['STATUS'].isin(['1', '2', '3', '4', '5'])).astype('int8')
    balance['BUREAU_DPD_STATUS'] = balance['STATUS']. \
        apply(lambda x: int(x) if x in ['1', '2', '3', '4', '5'] else 0)
    max_dpd = balance.groupby('SK_ID_BUREAU')[
        ['BUREAU_IS_DPD', 'BUREAU_DPD_STATUS']].agg('max').reset_index()
    data = data.merge(max_dpd, on=["SK_ID_BUREAU"], how="left")
    ori_feature = set(data.columns)
    data = BureauTransform(data)
    ext_feature = set(data.columns) - ori_feature
    df_groupby = data.groupby(["SK_ID_CURR"])
    df = df_groupby.size().reset_index().rename(columns={0: "Nums_Bureaus"})
    vc_feature = [
        'CREDIT_ACTIVE',
        'CREDIT_CURRENCY',
        'CREDIT_TYPE'
    ]
    for _f in vc_feature:
        df[f"BUREAU_{_f}_Nunique"] = df_groupby[_f].nunique()
    cat_feature = [
        'CREDIT_ACTIVE_IS_ACTIVE',
        'CREDIT_ACTIVE_IS_BAD',
        'CREDIT_ACTIVE_IS_CLOSED',
        'CREDIT_ACTIVE_IS_SOLD',
        'CREDIT_TYPE_IS_REVOLVING',
        'BUREAU_IS_DPD'
    ]
    for _f in cat_feature:
        df[f"BUREAU_{_f}_SUM"] = df_groupby[_f].sum()
    num_feature = {
        'AMT_ANNUITY': ['sum'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ["max", "sum"],
        'AMT_CREDIT_SUM_DEBT': ["max", "sum"],
        'AMT_CREDIT_SUM_LIMIT': ["max", "mean"],
        'AMT_CREDIT_SUM_OVERDUE': ["max", "mean"],
        'CNT_CREDIT_PROLONG': ["max", "mean"],
        'CREDIT_DAY_OVERDUE': ["max", "mean"],
        'DAYS_CREDIT': ["max", "mean"],
        'DAYS_CREDIT_ENDDATE': ["max", "mean"],
        'DAYS_CREDIT_UPDATE': ["max"],
        'BUREAU_DPD_STATUS': ["max", "mean"]
    }
    for _f, _aggs in num_feature.items():
        for _agg in _aggs:
            df[f"BUREAU_{_f}_{_agg}"] = df_groupby[_f].agg(_agg)
    print(df.shape)
    return df


def CreditTransform(data):
    df = data.copy()
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
    df.loc[df['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
    df.loc[df['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan

    # cat_feature
    df['NAME_CONTRACT_IS_COMPLETED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Completed').astype('int8')
    df['NAME_CONTRACT_IS_APPROVED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Approved').astype('int8')
    df['NAME_CONTRACT_IS_REFUESD'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int8')
    #     df['NAME_CONTRACT_IS_PROPOSAL'] = \
    #         (df['NAME_CONTRACT_STATUS']=='Sent proposal').astype('int8')
    #     df['NAME_CONTRACT_IS_DEMAND'] = \
    #         (df['NAME_CONTRACT_STATUS']=='Demand').astype('int8')
    #     df['NAME_CONTRACT_IS_SIGNED'] = \
    #         (df['NAME_CONTRACT_STATUS']=='Signed').astype('int8')

    df['NAME_BALANCE_IS_POSITIVE'] = \
        (df['AMT_BALANCE'] > 0).astype('int8')
    df['NAME_BALANCE_LT_PAYMENT'] = \
        (df['AMT_BALANCE'] > df['AMT_PAYMENT_TOTAL_CURRENT']).astype('int8')
    df['NAME_BALANCE_LT_MIN_PAYMENT'] = \
        (df['AMT_BALANCE'] > df['AMT_INST_MIN_REGULARITY']).astype('int8')
    df['NAME_RECEIVE_LT_MIN_PAYMENT'] = \
        (df['AMT_RECEIVABLE_PRINCIPAL'] > df['AMT_INST_MIN_REGULARITY']).astype('int8')
    df['NAME_RECEIVE_LT_PAYMENT'] = \
        (df['AMT_RECEIVABLE_PRINCIPAL'] > df['AMT_PAYMENT_TOTAL_CURRENT']).astype('int8')
    df['NAME_PAYMENT_LT_REGULATION'] = \
        (df['AMT_PAYMENT_CURRENT'] > df['AMT_INST_MIN_REGULARITY']).astype('int8')
    df['NAME_PAYMENT_LT_TOTAL'] = \
        (df['AMT_PAYMENT_CURRENT'] > df['AMT_PAYMENT_TOTAL_CURRENT']).astype('int8')
    df['NAME_CREDIT_LIMIT_IS_NONE'] = \
        (df['AMT_CREDIT_LIMIT_ACTUAL'] == 0).astype('int8')
    df['NAME_CREDIT_LIMIT_IS_NONE'] = \
        (df['AMT_CREDIT_LIMIT_ACTUAL'] == 0).astype('int8')

    df['AMT_RATIO_BALANCE_CREDIT'] = \
        df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL']
    df['AMT_RATIO_BALANCE_PAYMENT'] = \
        df['AMT_BALANCE'] / df['AMT_PAYMENT_TOTAL_CURRENT']
    df['AMT_RATIO_BALANCE_MIN_REGULATION'] = \
        df['AMT_BALANCE'] / df['AMT_INST_MIN_REGULARITY']
    df['AMT_RATIO_BALANCE_RECEIVABLE'] = \
        df['AMT_BALANCE'] / df['AMT_RECEIVABLE_PRINCIPAL']

    df['AMT_RATIO_PAYMENT_MIN'] = \
        df['AMT_INST_MIN_REGULARITY'] / df['AMT_PAYMENT_CURRENT']
    df['AMT_RATIO_MIN_TOTAL'] = \
        df['AMT_INST_MIN_REGULARITY'] / df['AMT_PAYMENT_TOTAL_CURRENT']
    df['AMT_RATIO_PAYMENT_TOTAL'] = \
        df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_PAYMENT_CURRENT']
    df['AMT_RATIO_PAYMENT_REVEIVABLE'] = \
        df['AMT_PAYMENT_CURRENT'] / df['AMT_RECEIVABLE_PRINCIPAL']
    df['AMT_RATIO_MIN_RECEIVABLE'] = \
        df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_RECEIVABLE_PRINCIPAL']
    df['AMT_RATIO_TOTAL_RECEIVABLE'] = \
        df['AMT_INST_MIN_REGULARITY'] / df['AMT_RECEIVABLE_PRINCIPAL']
    return df


def PaymentsTransform(data):
    df = data.copy()
    df['PAYMENT_OVERDUE'] = \
        (df['DAYS_ENTRY_PAYMENT'] > df['DAYS_INSTALMENT']).astype('int64')
    df['PAYMENT_NOT_ENOUGH'] = \
        (df['AMT_PAYMENT'] < df['AMT_INSTALMENT']).astype('int64')
    df['PAYMENT_FOR_CREDIT'] = \
        (df['NUM_INSTALMENT_VERSION'] == 0).astype('int64')
    # num_feature
    df['RATIO_PAYMENT_ENTRY'] = \
        df['DAYS_ENTRY_PAYMENT'] / df['DAYS_INSTALMENT']
    df['RATIO_PAYMENT_INSTALMENT'] = \
        df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
    return df


def PaymentsFeatureExtract(data):
    ori_feature = set(data.columns)
    data = PaymentsTransform(data)
    ext_feature = set(data.columns) - ori_feature
    df_groupby = data.groupby(["SK_ID_CURR", "SK_ID_PREV"])
    df_prev = df_groupby.size().reset_index().rename(
        columns={0: 'INSTL_NUMS_OF_INSTALMENT_VERSION'})
    print(df_prev.shape)
    df_prev['INSTAL_NUMS_OF_INSTALMENTS'] = \
        df_groupby['NUM_INSTALMENT_NUMBER'].max().values
    df_prev['INSTAL_NUMS_OF_INSTALMENTS_VERSION'] = \
        df_groupby['NUM_INSTALMENT_VERSION'].nunique().values
    for _f in data.columns[:4]:
        df_prev[f"PREV_{_f}_Mean"] = df_groupby[_f].mean().values

    all_feature = df_prev.columns
    df_groupby = df_prev.groupby(["SK_ID_CURR"])
    df = df_groupby.size().reset_index().rename(columns={0: 'POS_NUMS_OF_CREDITS'})
    for _f in all_feature:
        df[f"CREDIT_{_f}_max"] = df_groupby[_f].max()
    for _f in all_feature:
        df[f"CREDIT_{_f}_mean"] = df_groupby[_f].mean()
    print(df.shape)
    return df

def CreditFeatureExtract(data):
    ori_feature = set(data.columns)
    data = CreditTransform(data)
    ext_feature = set(data.columns) - ori_feature
    df_groupby = data.groupby(["SK_ID_CURR", "SK_ID_PREV"])
    df_prev = df_groupby.size().reset_index().rename(
        columns={0: "CREDIT_NUMS_OF_MONTHS_BALANCE_RECORD"})
    print(df_prev.shape)
    for _f in list(ext_feature)[-12:]:
        df_prev[f"PREV_{_f}_Sum"] = df_groupby[_f].sum().values
    num_feature = [
        'AMT_BALANCE',
        'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_DRAWINGS_CURRENT',
        'AMT_INST_MIN_REGULARITY',
        'AMT_PAYMENT_CURRENT',
        'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECIVABLE',
        'CNT_DRAWINGS_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL',
        'CNT_INSTALMENT_MATURE_CUM',
        'SK_DPD_DEF', 'SK_DPD']
    for _f in list(ext_feature)[:-12] + num_feature:
        df_prev[f"BUREAU_{_f}_Sum"] = df_groupby[_f].mean().values
    all_feature = df_prev.columns
    mean_feature = list(set(df_prev.columns) - set(df_prev.columns[1:13]))
    df_groupby = df_prev.groupby(["SK_ID_CURR"])
    df = df_groupby.size().reset_index().rename(columns={0: 'CREDIT_NUMS_OF_CREDITS'})
    for _f in all_feature:
        df[f"CREDIT_{_f}_max"] = df_groupby[_f].max()
    for _f in mean_feature:
        df[f"CREDIT_{_f}_mean"] = df_groupby[_f].mean()
    print(df.shape)
    return df


def PosCashTransform(data):
    df = data.copy()
    df['CONTRACT_STATUS_IS_REFUSED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int8')
    df['CONTRACT_STATUS_IS_CANCEL'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Canceled').astype('int8')
    df['CONTRACT_STATUS_IS_SIGN'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Signed').astype('int8')
    df['CONTRACT_STATUS_IS_APPROVED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Approved').astype('int8')
    df['CONTRACT_STATUS_IS_RETURNED'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Returned to the store').astype('int8')
    df['CONTRACT_STATUS_IS_DEBT'] = \
        (df['NAME_CONTRACT_STATUS'] == 'Amortized debt').astype('int8')

    df['SK_DPD_IS_POSITIVE'] = \
        (df['SK_DPD'] > 0).astype('int8')
    df['SK_DPD_DEF_IS_POSITIVE'] = \
        (df['SK_DPD_DEF'] > 0).astype('int8')
    df['CNT_INSTALMENT_FUTURE_IS_POSITIVE'] = \
        (df['CNT_INSTALMENT_FUTURE'] > 0).astype('int8')
    # num_feature
    df['CNT_INSTALMENT_FUTURE_PERCENTAGE'] = \
        df['CNT_INSTALMENT'] / df['CNT_INSTALMENT_FUTURE']
    return df


def PosCashFeatureExtract(data):
    ori_feature = set(data.columns)
    data = PosCashTransform(data)
    ext_feature = set(data.columns) - ori_feature
    df_groupby = data.groupby(["SK_ID_CURR", "SK_ID_PREV"])
    df_prev = df_groupby.size().reset_index().rename(
        columns={0: 'POS_NUMS_OF_MONTHS_BALANCE_RECORD'})
    df_prev['POS_NUMS_OF_MONTHS'] = \
        df_groupby['MONTHS_BALANCE'].first().abs().values
    df_prev['POS_NUMS_OF_INSTALMENT'] = \
        df_groupby['CNT_INSTALMENT'].nunique().values
    vc_feature = data.columns[8:14]
    last_feature = [
        'CNT_INSTALMENT_FUTURE',
        'CNT_INSTALMENT_FUTURE_IS_POSITIVE',
        'CNT_INSTALMENT_FUTURE_PERCENTAGE'
    ]
    max_feature = [
        'SK_DPD',
        'SK_DPD_DEF',
        'CNT_INSTALMENT',
        'CNT_INSTALMENT_FUTURE_PERCENTAGE'
    ]

    for _f in vc_feature:
        df_prev[f"PREV_{_f}_Sum"] = df_groupby[_f].sum().values
    for _f in last_feature:
        df_prev[f"PREV_{_f}_Last"] = df_groupby[_f].last().values
    for _f in max_feature:
        df_prev[f"PREV_{_f}_Max"] = df_groupby[_f].max().values
    for _f in data.columns[:4]:
        df_prev[f"PREV_{_f}_Mean"] = df_groupby[_f].mean().values

    all_feature = df_prev.columns
    df_groupby = df_prev.groupby(["SK_ID_CURR"])
    df = df_groupby.size().reset_index().rename(columns={0: 'POS_NUMS_OF_CREDITS'})
    for _f in all_feature:
        df[f"CREDIT_{_f}_max"] = df_groupby[_f].max()
    for _f in all_feature:
        df[f"CREDIT_{_f}_mean"] = df_groupby[_f].mean()
    print(df.shape)
    return df


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/HomeCredit"):
        os.system("kaggle competitions download -c home-credit-default-risk -p data")
        unzip_file("data/home-credit-default-risk.zip", "data/HomeCredit")
    df_application = pd.read_csv("data/HomeCredit/application_train.csv").sort_values(["SK_ID_CURR"])
    df_application = ApplicationFeatureExtract(df_application)
    sk_id_curr_cate = set(df_application["SK_ID_CURR"])

    df_previous = pd.read_csv("data/HomeCredit/previous_application.csv")
    df_previous = df_previous.loc[df_previous["SK_ID_CURR"].isin(sk_id_curr_cate)].\
        sort_values(["SK_ID_CURR", "SK_ID_PREV"])
    df_previous = PreviousFeatureExtract(df_previous)

    df_bureau = pd.read_csv("data/HomeCredit/bureau.csv")
    df_bureau_bal = pd.read_csv("data/HomeCredit/bureau_balance.csv")
    df_bureau = df_bureau.loc[df_bureau["SK_ID_CURR"].isin(sk_id_curr_cate)]. \
        sort_values(["SK_ID_CURR", "SK_ID_BUREAU"])
    df_bureau_bal = df_bureau_bal.loc[df_bureau_bal["SK_ID_BUREAU"].isin(set(df_bureau["SK_ID_BUREAU"]))].\
        sort_values(["SK_ID_BUREAU"])
    df_bureau = BuearuFeatureExtract(df_bureau, df_bureau_bal)
    del df_bureau_bal
    gc.collect()

    df_credit = pd.read_csv("data/HomeCredit/credit_card_balance.csv")
    df_credit = df_credit.loc[df_credit["SK_ID_CURR"].isin(sk_id_curr_cate)].\
        sort_values(["SK_ID_CURR", "SK_ID_PREV"])
    df_credit = CreditFeatureExtract(df_credit)

    df_payments = pd.read_csv("data/HomeCredit/installments_payments.csv")
    df_payments = df_payments.loc[df_payments["SK_ID_CURR"].isin(sk_id_curr_cate)].\
        sort_values(["SK_ID_CURR", "SK_ID_PREV"])
    df_payments = PaymentsFeatureExtract(df_payments)

    df_pos_cash = pd.read_csv("data/HomeCredit/pos_cash_balance.csv")
    df_pos_cash = df_pos_cash.loc[df_pos_cash["SK_ID_CURR"].isin(sk_id_curr_cate)]. \
        sort_values(["SK_ID_CURR", "SK_ID_PREV"])
    df_pos_cash = PosCashFeatureExtract(df_pos_cash)

    with open("data/HomeCredit/FeatureExtract/df_appliaction", "wb") as f:
        pickle.dump(df_application, f)
    with open("data/HomeCredit/FeatureExtract/df_bureau", "wb") as f:
        pickle.dump(df_bureau, f)
    with open("data/HomeCredit/FeatureExtract/df_credit", "wb") as f:
        pickle.dump(df_credit, f)
    with open("data/HomeCredit/FeatureExtract/df_payments", "wb") as f:
        pickle.dump(df_payments, f)
    with open("data/HomeCredit/FeatureExtract/df_pos_cash", "wb") as f:
        pickle.dump(df_pos_cash, f)
    with open("data/HomeCredit/FeatureExtract/df_previous", "wb") as f:
        pickle.dump(df_previous, f)



