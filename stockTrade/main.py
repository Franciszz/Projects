# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:10:42 2018

@author: chenyi
"""

import sys
sys.path.append('src')
import pickle
from process_input import JzStockData
from process_stock_detection import JzStockDetection
from process_stock_position import JzStockPosition
from process_daily_feature import JzDailyFeature
from process_daily_netasset import JzNetAsset
""" 临时储存数据 """ 
def from_pickle(filename):
    with open('cache/'+filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
def to_pickle(obj, filename):
    with open('cache/'+filename, 'wb') as f:
        pickle.dump(obj, f, -1)
        
""" 数据全体 """
inputProcessor = JzStockData()
jz_data = inputProcessor.transform()
#jz_data.to_csv('result/jz_data.csv',index=False)
#to_pickle(jz_data,'jz_data.csv')
#jz_data = from_pickle('jz_data.csv')

""" 股票数据 """ 
jz_stock = inputProcessor.jz_stock_generate(jz_data)
#jz_data.to_csv('result/jz_stock.csv',index=False)
#to_pickle(jz_stock,'jz_stock.csv')
#jz_stock = from_pickle('jz_stock.csv')

""" 股票交易数据 """ 
detectionProcessor = JzStockDetection()
jz_daily_stock_flow = detectionProcessor.daily_flow_generate(jz_stock)
#jz_daily_stock_flow.to_csv('result/jz_daily_stock_flow.csv',index=False)
#to_pickle(jz_daily_stock_flow,'jz_daily_stock_flow.csv')

""" 银证转帐数据与基金买卖数据 """ 
jz_daily_transfer = detectionProcessor.daily_transfer_inout(jz_data)
#jz_daily_transfer.to_csv('result/jz_daily_transfer.csv',index=False)
#to_pickle(jz_daily_transfer,'jz_daily_transfer.csv')

""" 股票数据每日汇总 """ 
jz_daily_stock = detectionProcessor.stock_trade_detail(jz_daily_stock_flow)
#jz_daily_stock.to_csv('result/jz_daily_stock.csv',index=False)
#to_pickle(jz_daily_stock,'jz_daily_stock.csv')

""" 股票数据修正与首日持股检测 """  
jz_daily_stock_revise = detectionProcessor.stock_extra_detection(jz_daily_stock)
#jz_daily_stock_revise.to_csv('result/jz_daily_stock_revise.csv',index=False)
#to_pickle(jz_daily_stock_revise,'jz_daily_stock_revise.csv')
#colOrders = jz_daily_stock.columns.tolist()[:10]+['cash']
#jz_daily_stock_revise = jz_daily_stock_revise.reindex(columns = colOrders)
#jz_daily_stock_revise = from_pickle('jz_daily_stock_revise.csv') 

""" 每日持仓与累计盈亏计算 """
stockPositionProcessor = JzStockPosition()
jz_daily_stock_justify = stockPositionProcessor.daily_stock_ambulance_justify(
        jz_daily_stock_revise)
jz_daily_stock_split = stockPositionProcessor.daily_stock_split(
        jz_daily_stock_justify)
jz_stock_position_inland = stockPositionProcessor.stock_position_generate(
        jz_daily_stock_split['jz_daily_stock_inland'])
jz_stock_position_inland = from_pickle('jz_stock_position_inland.csv')
jz_stock_position_offland = stockPositionProcessor.stock_position_generate(
        jz_daily_stock_split['jz_daily_stock_offland'])

#jz_stock_position_inland.to_csv('result/jz_stock_position_inland.csv',index=False)
#to_pickle(jz_stock_position_inland,'jz_stock_position_inland.csv')
#jz_stock_position_offland.to_csv('result/jz_stock_position_offland.csv',index=False)
#to_pickle(jz_stock_position_offland,'jz_stock_position_offland.csv')
#jz_stock_position_offland = from_pickle('jz_stock_position_offland.csv')

""" 用户交易特征 """
dailyFeatureProcessor = JzDailyFeature()
jz_daily_feature_offland = dailyFeatureProcessor.daily_feature_element_concat(
        jz_stock_position_offland, jz_daily_stock_justify)
jz_daily_feature_inland = dailyFeatureProcessor.daily_feature_element_concat(
        jz_stock_position_inland, jz_daily_stock_justify)
#jz_daily_feature_inland.to_csv('result/jz_daily_feature_inland.csv',index=False)
#to_pickle(jz_daily_feature_inland,'jz_daily_feature_inland.csv')
#jz_daily_feature_offland.to_csv('result/jz_daily_feature_offland.csv',index=False)
#to_pickle(jz_daily_feature_offland,'jz_daily_feature_offland.csv')
#jz_daily_feature_offland = from_pickle('jz_daily_feature_offland.csv')
#jz_daily_feature_inland = from_pickle('jz_daily_feature_inland.csv')
#jz_daily_feature_inland = from_pickle('jz_daily_feature_inland.csv')

""" 银证转帐数据修正及总资产 """
netAssetProcessor = JzNetAsset()
jz_daily_netAsset_offland = netAssetProcessor.jz_netasset_generate(
        jz_daily_feature_offland, jz_data)
jz_daily_netAsset_inland = netAssetProcessor.jz_netasset_generate(
        jz_daily_feature_inland, jz_data)

#jz_daily_netAsset_inland.to_csv('result/jz_daily_netAsset_inland.csv',index=False)
#to_pickle(jz_daily_netAsset_inland,'jz_daily_netAsset_inland.csv')
#
#jz_daily_netAsset_offland.to_csv('result/jz_daily_netAsset_offland.csv',index=False)
#to_pickle(jz_daily_netAsset_offland,'jz_daily_netAsset_offland.csv')
