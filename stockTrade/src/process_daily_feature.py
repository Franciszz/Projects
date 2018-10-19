# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:40:15 2018

@author: chenyi
"""
import numpy as np
from process_stock_position import JzStockPosition

class JzDailyFeature(JzStockPosition):
    
    def __init__(self):
        super(JzDailyFeature,self).__init__()
    
    def daily_feature_element_concat(self, jz_stk_pos, jz_daily_stk):
        stock_position_df = jz_stk_pos.copy()
        price_df = self.stock_mkt_price_generate()
        stock_position_df = stock_position_df.merge(
                price_df, on=['deliveryDate','securitiesCode'], how='left')
        stock_position_df['price'] = stock_position_df['price'].fillna(0)
        stock_position_df['mktValue'] = \
            stock_position_df['securitiesBalance']*stock_position_df['price']
        stock_position_df['mktValueProdSecurityDays'] = \
            stock_position_df['securitiesDays']*stock_position_df['mktValue']
        groupby_stkpos_df = stock_position_df.groupby(['fundsAccount','deliveryDate']) 
        """ 当日持仓市值 """
        feature = groupby_stkpos_df['mktValue'].sum().reset_index()
        """ 当日是否空仓 """
        feature['isEmpty'] = (feature['mktValue']==0).astype('int32')
        """ 累计利润 """
        component = groupby_stkpos_df['securityProfit'].first().reset_index()
        feature = feature.merge(component, on=['fundsAccount','deliveryDate'],how='left')
        """ 当日持股代码 """
        component = groupby_stkpos_df['securitiesCode'].apply(lambda x: set(x)).reset_index().\
            rename(columns = {'securitiesCode':'listSecuritiesCode'})
        feature = feature.merge(component,on=['fundsAccount','deliveryDate'],how='left')
        feature['listSecuritiesCode'] = feature['listSecuritiesCode'].map(
                lambda x: x-set(['000000']))
        """ 当日持股数目 """
        feature['numSecurity'] = feature['listSecuritiesCode'].map(len)
        """ 当日持有中小盘股票数目 """
        feature['numMidSmallSecurity'] = feature['listSecuritiesCode'].map(
                lambda x: np.array(list(map(lambda y:(y.startswith('002'))&(len(x)==6),x))).sum())
        """ 当日持有创业板股票数目 """
        feature['numChiNextSecurity'] = feature['listSecuritiesCode'].map(
                lambda x: np.array(list(map(lambda y:y.startswith('300'),x))).sum())
        feature['numPocketSecurity'] = \
            feature['numMidSmallSecurity'] + feature['numChiNextSecurity']
        """ 当日加权持仓天数 """
        component = groupby_stkpos_df['mktValueProdSecurityDays'].sum().reset_index().\
            rename(columns={'mktValueProdSecurityDays':'securityDaysWeighted'})
        feature = feature.merge(component, on=['fundsAccount','deliveryDate'],how='left')
        
        daily_stock_df = jz_daily_stk[[
                'fundsAccount','deliveryDate','securitiesCode','funds_sell','funds_buy','cash',
                'securityDays','securitySellOut']].rename(
                columns={'securityDays':'numBuildPositionAverageMonthly',
                         'securitySellOut':'numSellOutAverageMonthly'})
        daily_stock_df['numSellOutAverageMonthly'] = \
            daily_stock_df['numSellOutAverageMonthly'].fillna(0)    
        groupby_dailystk_df = daily_stock_df.groupby(['fundsAccount','deliveryDate'])
        """ 当日持有现金 """
        component = groupby_dailystk_df['cash'].first().reset_index()
        feature = feature.merge(component,on=['fundsAccount','deliveryDate'],how='left')
        feature['cash'] = feature['cash'].fillna(method='ffill')
        """ 单日总资产 """
        feature['totalAsset'] = feature['mktValue']+feature['cash']
        feature['securityDaysWeighted'] = \
            round(feature['securityDaysWeighted']/feature['totalAsset'],2)
        """ 仓位 """
        feature['securityRate'] = round(feature['mktValue']/feature['totalAsset']*100,2)
        """ 当日股票买入卖出额度 """
        component = groupby_dailystk_df[[
                'funds_sell','funds_buy','numBuildPositionAverageMonthly',
                'numSellOutAverageMonthly']].sum().abs().reset_index()
        feature = feature.merge(component,on=['fundsAccount','deliveryDate'],how='left')
        """ 资产周转率 """
        feature['assetTurnoverRate'] = round((
                feature['funds_sell']+feature['funds_buy'])/feature['totalAsset']/2*100,2)
        feature[['funds_sell','funds_buy','numBuildPositionAverageMonthly',
                 'numSellOutAverageMonthly','assetTurnoverRate']] = \
            feature[['funds_sell','funds_buy','numBuildPositionAverageMonthly',
                     'numSellOutAverageMonthly','assetTurnoverRate']].fillna(0)
            
        groupby_feature = feature.groupby(['fundsAccount'])
        """ 过去30日平均持股个数 """
        feature['numSecurityAverageMonthly'] = \
            groupby_feature['numSecurity'].\
            rolling(30,min_periods=1).mean().values
        """ 过去30日开仓次数 """
        feature['numBuildPositionAverageMonthly'] = \
            groupby_feature['numBuildPositionAverageMonthly'].\
            rolling(30,min_periods=1).mean().values
        """ 过去30日清仓次数 """
        feature['numSellOutAverageMonthly'] = \
            groupby_feature['numSellOutAverageMonthly'].\
            rolling(30,min_periods=1).mean().values
        """ 过去30日空仓天数 """
        feature['numEmptyAverageMonthly'] = \
            groupby_feature['isEmpty'].rolling(30,min_periods=1).sum().values
        """ 过去30日平均资产周转率 """
        feature['assetTurnoverRateAverageMonthly'] = \
            groupby_feature['assetTurnoverRate'].rolling(30,min_periods=1).mean().values
        """ 过去30日平均持股比例 """
        feature['securityRateAverageMonthly'] = \
            groupby_feature['securityRate'].rolling(30,min_periods=1).mean().values
        """ 过去30日持有的不同股票的数目 """
        return feature
        
