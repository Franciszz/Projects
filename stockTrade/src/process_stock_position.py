# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:39:51 2018

@author: chenyi
"""

import pandas as pd
import numpy as np

from copy import deepcopy
from datetime import timedelta

class JzStockPosition(object):
    
    def __init__(self):
        super(JzStockPosition, self).__init__()
    
    def stock_mkt_price_generate(self):
        """ 导入价格与汇率数据 """
        price_df = pd.read_csv('data/stk_mkt_price.csv')
        price_df.columns = ['securitiesCode']+pd.to_datetime(price_df.columns[1:]).tolist()
        price_df['securitiesCode'] = price_df['securitiesCode'].map(
            lambda x: '0'+x if len(x)==7 else x)
        price_df = price_df.set_index('securitiesCode').T.\
            reindex(pd.date_range('2014-01-01','2018-07-13')).\
            fillna(method='ffill').T.reset_index(drop=False).\
            pipe(pd.melt,id_vars=['securitiesCode']).sort_values(['securitiesCode','variable']).\
            reset_index(drop=True).rename(columns={'variable':'deliveryDate','value':'price'})
        price_df['is_offland'] = price_df['securitiesCode'].str.endswith('HK').astype('int64')
        current_rate_df = pd.read_csv('data/current.csv', parse_dates=[0]).\
            rename(columns={'ratedate':'deliveryDate'})
        price_df = price_df.merge(current_rate_df,on=['deliveryDate'],how='left')
        """ 非交易日股票价格以前一交易日收盘价填充，首日价格以第二日填充 """
        price_df['price'] = price_df['price'].fillna(method='bfill')
        price_df['rate'] = price_df['rate'].fillna(method='ffill')
        """ 港股价格乘以当日汇率 """
        price_df['price'] = np.where(price_df['is_offland']==1,
                price_df['price']*price_df['rate'],
                price_df['price'])
        price_df.drop(['rate','is_offland'],axis=1,inplace=True)
        price_df['securitiesCode'] = price_df['securitiesCode'].map(lambda x: str(x)[:-3])
        return price_df
    
    def daily_stock_ambulance_justify(self, data):
        data = data.copy()
        price_df = self.stock_mkt_price_generate()
        data = data.merge(price_df, on=['deliveryDate','securitiesCode'], how='left')
        """ 上市新股修正价格 """
        data['price'] = np.where(data['isNew']==1,data['price']/1.44,data['price'])
        
        data = data.reindex(columns = data.columns.tolist()+[
                'securityMarketValue','securityDays','securitySellOut']).\
            assign(securityMarketValue = data['price']*data['securityBalance'])
        data_fst = data[['fundsAccount','deliveryDate']].groupby(['fundsAccount'],as_index=False).\
            first().assign(isFirst=1)
        data = data.merge(data_fst, on = ['fundsAccount','deliveryDate'], how='left')
        """ 确定初始交易日及其成本 """
        data['extra_cost'] = np.where(
                data['isFirst'].notnull(),
                -data['securityMarketValue'],
                data['extra_cost'])
        """ 确定每笔交易是否为开仓 """
        data['securityDays'] = np.where(
                (data['extra_number']==data['securityBalance'])&(data['number_sell']==0),1,0)
        """ 确定每笔交易是否为清仓 """
        data['securitySellOut'] = np.where(
                (data['number_buy']==0)&(data['securityBalance']==0),1,np.nan)
        """ 处理转托管出入/指定出入/新股的现金流 """
        condition_sell = (data['number_sell']>0)&(data['funds_sell']==0)
        condition_buy = (data['number_buy']>0)&(data['funds_buy']==0)
        data['funds_sell'] = np.where(
                condition_sell,data['number_sell']*data['price'],data['funds_sell'])
        data['funds_buy'] = np.where(
                condition_buy,-data['number_buy']*data['price'],data['funds_buy'])
        data['extra_cost'] = np.where(
                condition_buy|condition_sell,data['funds_buy']+data['funds_sell'],data['extra_cost'])
    #    data['cash'] = np.where(
    #            condition_buy|condition_sell,data['cash']+data['extra_cost'],data['cash'])
    #    data = data.groupby(['fundsAccount','securitiesCode']).apply(daily_record_update)
        return data
    
#    def daily_record_update(self,data):
#        data = data.copy()
#        data['securityBuildPosition'] = np.where(
#                (data['extra_number']==data['securityBalance'])&(data['number_sell']==0),1,0)
#        data['securityBuildPosition'] = data['securityBuildPosition'].cumsum()
#        return data  
        """ 有些帐户有指定入账的交易未处理 """
    def daily_stock_split(self, data):
        data = data.copy()
        category_fundsAccount_hk = \
            data[data['securitiesCode'].str.len()==5]['fundsAccount'].unique().tolist()
        inland_data = data[~(data['fundsAccount'].isin(category_fundsAccount_hk))]
        offland_data = data[data['fundsAccount'].isin(category_fundsAccount_hk)]
        return {'jz_daily_stock_inland': inland_data,
                'jz_daily_stock_offland': offland_data}
        
    def stock_position_generate(self, data):
        data = data.copy()
        data = data[['fundsAccount','deliveryDate','securitiesCode','securityBalance',
                     'extra_cost','securityDays']].rename(
                     columns={'extra_cost':'securityCost'})
        data['securityCost'] = -data['securityCost']
        data = data.groupby(['fundsAccount']).\
            apply(self.stock_position_individual).reset_index(drop=True)
        return data  
    
    def stock_position_individual(self, data):
        data = data.copy()
        data = data.sort_values(['deliveryDate','securitiesCode']).reset_index(drop=True)
        data['securityDays'] = 1
        """ 新建客户持仓字典 """ 
        stock_position_dict = {}
        """ 新建客户当日实现利润字典 """
        profit_dict = {}
        """ 获取帐户名 """
        fundsAccount = data['fundsAccount'].tolist()[0]
        print(fundsAccount)
        """ 新建客户持仓字前一日字典 """
        stock_position_dict_former = {}
        """ 将单客户数据依据交易日进行分组并获取第一天持仓 """
        groupby_deliveryDate_data = data.groupby('deliveryDate')
        groups_deliveryDate_category = data['deliveryDate'].tolist()  
        """ 确定初始交易日的持仓及以当日价格作为初始成本  """
        deliveryDate_former = groups_deliveryDate_category[0]
        deliveryDate_former_df = groupby_deliveryDate_data.get_group(deliveryDate_former)
        profit_dict[deliveryDate_former] = 0
#       record = ['fundsAccount','deliveryDate','securitiesCode','securityBalance',
#                 'extra_cost','securityDays']
        for record in deliveryDate_former_df.values.tolist():
            stock_position_dict_former[record[2]] = record[3:]
        stock_position_dict[deliveryDate_former] = deepcopy(stock_position_dict_former)
        while deliveryDate_former < pd.to_datetime('2018-04-02'):
            """ 交易日期往后加一天，记为第二日 """
            deliveryDate_latter = deliveryDate_former + timedelta(days=1)
            """ 新建客户持仓字后一日字典 """
            stock_position_dict_latter = {}
            profit_dict[deliveryDate_latter] = profit_dict[deliveryDate_former]
            if deliveryDate_latter in groups_deliveryDate_category:
                """如果后一日在数据中，则进行以下操作 """
                deliveryDate_latter_df = groupby_deliveryDate_data.get_group(deliveryDate_latter)
                security_set_former = set(stock_position_dict_former.keys())-set(['000000'])
                security_set_implement = set(security_set_former) - set(deliveryDate_latter_df.securitiesCode)
                for record in deliveryDate_latter_df.values.tolist():
                    """ 后一日当天有交易的股票如果在前一日的股票列表中，且证券余额大于0，则总成本相加，持股天数+1 """
                    """ 若证券余额等于0，不做记录，加入账户当日盈利字典 """
                    """ 后一日有交易的股票如果不在前一日的股票列表中，为新建仓股票，持股天数记为1 """
                    if record[2] in security_set_former:
                        record[4] += stock_position_dict_former[record[2]][1]
                        if record[3] > 0:
                            record[5] = stock_position_dict_former[record[2]][2] + 1
                            stock_position_dict_latter[record[2]] = record[3:]
                        else:
                            profit_dict[deliveryDate_latter] += -record[4]#record[-1] - date_pos_fst[record[0]][1]
                    else:  
                        stock_position_dict_latter[record[2]] = record[3:]
                if len(security_set_implement) > 0:
                    """ 前一日的股票列表中在后一日未交易，其他不变，持股天数加1 """                
                    for securityCode in security_set_implement:
                        days = stock_position_dict_former[securityCode][2]+1
                        stock_position_dict_latter[securityCode] = \
                            stock_position_dict_former[securityCode][:2]+[days]
                if len(stock_position_dict_latter.keys())==0:
                    """ 若空仓则记录为000000 """ 
                    stock_position_dict_latter['000000'] = [0,0,0]
            else:
                for securityCode in stock_position_dict_former.keys():
                    """ 后一日无交易，其他不变，持股天数+1 """                
                    days = stock_position_dict_former[securityCode][2]+1
                    stock_position_dict_latter[securityCode] = \
                        stock_position_dict_former[securityCode][:2]+[days]
            stock_position_dict[deliveryDate_latter] = stock_position_dict_latter
            stock_position_dict_former = deepcopy(stock_position_dict_latter)
            deliveryDate_former = deliveryDate_latter 
        """将字典转数据框"""
        stock_position_list=[]
        for deliveryDate in stock_position_dict.keys():
            stock_position_deliveryDate_dict = stock_position_dict[deliveryDate]
            for securityCode in stock_position_deliveryDate_dict.keys():
                stock_position_list.append([
                        deliveryDate,securityCode]+stock_position_deliveryDate_dict[securityCode])
        profit_df = pd.DataFrame(list(profit_dict.items()),columns=['deliveryDate','securityProfit'])
        data = pd.DataFrame(stock_position_list, columns=[
                'deliveryDate','securitiesCode','securitiesBalance','securitiesCost','securitiesDays'])
        data.insert(0,'fundsAccount',fundsAccount)
        data = data.merge(profit_df,on=['deliveryDate'],how='left')
        return data