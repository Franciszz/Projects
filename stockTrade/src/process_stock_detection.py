# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:36:43 2018

@author: chenyi
"""

import pandas as pd
import numpy as np

from datetime import timedelta

class JzStockDetection(object):
    
    def __init__(self):
        super(JzStockDetection,self).__init__()
        
    def daily_flow_generate(self, data):
        data = data.copy()
        self.abstract_enzh_map = pd.read_table(
                'data/jz_stock_flow_daily_add.txt', sep=' ', usecols=[7,8], 
                encoding='utf-8').drop_duplicates().values
        self.abstract_name_category = [
                '证券卖出', '证券买入', 'Tn证券卖出', 'Tn证券买入', '港股通股票卖出', 
                '港股通股票买入', '转托管入', '转托管出', '指定入账', '股份转入', '股份转出',
                '新股入账', '红股入账']
        self.abstract_code_category = [
                x for (x,y) in zip(self.abstract_enzh_map[:,0], self.abstract_enzh_map[:,1]) \
                if y in self.abstract_name_category]+[220010, 220004, 220005, 221006]
        """ 判断是否为买入 """
        data['isBuy'] = data['abstractCode'].isin([
                220000, 220100, 220094, 221014, 220004, 220005, 220010, 220015]).astype('int64')
        """ 判断是否为新股 """
        data['isNew'] = (data['abstractCode'].isin([220004])).astype('int64')
#        """ 判断是否为股份转出转入 """
#        data['isTran'] = (data['abstractCode'].isin([220005,221006,221014,220015])).astype('int64')
        data['turnoverNumber'] = data['turnoverNumber'].astype('int64')
        data = data[data.abstractCode.isin(self.abstract_code_category)]\
                [['fundsAccount','deliveryDate','transactionTime','securitiesCode','isBuy','isNew',
                  'turnoverNumber','securityBalance','fundsNumber','fundBalance']].\
                  sort_values(['fundsAccount','deliveryDate','transactionTime','securitiesCode'])
        return data
    
    def daily_transfer_inout(self, data):
        data = data.copy()
        """ 获取银证转入转出，基金买入卖出资金情况"""
        data = data[data.abstractCode.isin([
                140021,140055,140057,140211,140212,160021,160022,168007,240509,
                240511,240514,220049,221049,221038,220041,220024,240510])]#240509为水星1号买入视为银证转出 
        data['abstractType'] = data['abstractCode'].apply(lambda x: 1 if x in [
                140055,140211,160021,168007,221038,221049,
                240511,240514] else 0)
        """ 计算每日总转入转出资金 """                    
        data = data[['fundsAccount','deliveryDate','fundsNumber','abstractType']].\
                groupby(['fundsAccount','deliveryDate','abstractType']).\
                agg({'fundsNumber':'sum'}).reset_index().\
                pipe(pd.pivot_table,values='fundsNumber',
                     index=['fundsAccount', 'deliveryDate'],
                     columns=['abstractType'],fill_value=0).\
                     reset_index().rename(columns={0:'TransferIn',1:'TransferOut'})
        return data

    def stock_trade_detail(self, data):
        data = data.copy()
        
        """ 逆推每笔交易之前的现金 """ 
        data['cash'] = data['fundBalance']-data['fundsNumber']
        
        """ 每日首笔现金 """
        self.cash_df = data.groupby(['fundsAccount','deliveryDate'])['cash'].first().reset_index()
        cash_shift = self.cash_df.groupby(['fundsAccount'])['cash'].shift(-1)
        
        """ 现金最后一条记录为缺失需要用最后一次交易的fundBalance填补缺失值 """
        cash_df = data.groupby(['fundsAccount','deliveryDate'])['fundBalance'].last().reset_index()
        cash_df['cash'] = np.where(cash_shift.notnull(),cash_shift,cash_df['fundBalance'])
        cash_df.drop(['fundBalance'],axis=1,inplace=True)
        
        """ 每日股票买卖数额与金额 """
        trade_df = data.groupby(['fundsAccount','deliveryDate', 'securitiesCode','isBuy'])\
            ['turnoverNumber','fundsNumber', 'isNew'].sum().reset_index().\
            pipe(pd.pivot_table, values=['turnoverNumber','fundsNumber'], 
                 index=['fundsAccount', 'deliveryDate', 'securitiesCode','isNew'],
                 columns=['isBuy'], fill_value=0).reset_index()
        trade_df.columns = ['fundsAccount','deliveryDate', 'securitiesCode','isNew',
                            'funds_sell','funds_buy','number_sell','number_buy']
        
        """ 获取每只股票每天净买入数量校准股票余额 """
        trade_df['extra_cost'] = trade_df['funds_buy']+trade_df['funds_sell']
        
        """ 获取每只股票每天净买入金额作为成本变化计算 """
        trade_df['extra_number'] = trade_df['number_buy']-trade_df['number_sell']
        
        """ 获取每只股票最后一笔交易后的股票余额 """
        secBal_df = data.sort_values([
                'fundsAccount','securitiesCode','deliveryDate','transactionTime']).\
            groupby(['fundsAccount', 'securitiesCode', 'deliveryDate'])['securityBalance'].\
            last().reset_index()
            
        """ 获取每只股票最近一个交易日发生交易后的股票余额 """
        secBal_df['securityBalanceLast'] = secBal_df.groupby(['fundsAccount','securitiesCode'])\
            ['securityBalance'].shift(1).fillna(0)
        trade_df = trade_df.merge(secBal_df,
                                  on=['fundsAccount','deliveryDate', 'securitiesCode'],how='left')
        trade_df = trade_df.merge(cash_df,
                                  on=['fundsAccount','deliveryDate'],how='left')
        return trade_df
    
    def stock_extra_detection(self, data):
        data = data.copy()
        colOrders = data.columns.tolist()[:3]+data.columns.tolist()[4:11]+['cash','isNew']    
        """ 计算每只股票当日交易前合理股票余额用于校准 """
        data['securityBalanceLastReasonable'] = data['securityBalance']-data['extra_number']
        
        """ 获取每个用户每只股票在数据中第一次交易记录 """
        detection_df = data.groupby(['fundsAccount','securitiesCode']).first().reset_index()
        
        """ 计算第一次交易记录之前已有股票余额并筛选大于0的股票作为首日持股 """
        detection_df = detection_df[detection_df['securityBalanceLastReasonable']>0].\
            sort_values(['fundsAccount','deliveryDate']).\
            rename(columns={'securityBalanceLastReasonable':'securityFirstPosition'})
        
        """ 尽可能修正原有数据时间戳的乱序问题(原始数据存在问题) """
        data = data.merge(detection_df[['fundsAccount','deliveryDate','securitiesCode',
                                        'securityFirstPosition']], 
            on=['fundsAccount','deliveryDate','securitiesCode'], how='left')

        """ 对于不是用户第一次交易记录的股票记录作修正 """
        ambulance_df = data.loc[
                (data['securityBalanceLast']!=data['securityBalanceLastReasonable'])&
                 (data['securityFirstPosition'].isnull()),:]
        ambulance_df = self.update_ambulance_securityBalance(ambulance_df)\
            [['fundsAccount','deliveryDate','securitiesCode','securityBalance']].\
            rename(columns={'securityBalance':'securityBalanceAmbulance'})
        data = data.merge(ambulance_df,
                          on=['fundsAccount','deliveryDate','securitiesCode'],how='left')
        data['securityBalance'] = np.where(data['securityBalanceAmbulance'].notnull(),
            data['securityBalanceAmbulance'], data['securityBalance'])
        data.drop(['securityBalanceLast','securityBalanceLastReasonable',
                   'securityFirstPosition','securityBalanceAmbulance'],axis=1,inplace=True)
        
        """ 返回首日持仓数据并加入初始数据中 """
        first_position_df = self.stock_first_position(detection_df)
        data = pd.concat([data,first_position_df],axis=0).\
            sort_values(['fundsAccount','deliveryDate','securitiesCode']).reset_index(drop=True)
        data = data.reindex(columns = colOrders)
        return data

    def update_ambulance_securityBalance(self, data):
        data = data.copy().sort_values(['fundsAccount','securitiesCode','deliveryDate'])
        
        """ 获取每个用户每只股票的第一条错误记录 """
        first_ambulance_df = data.groupby(['fundsAccount','securitiesCode'],as_index=False).first()
        
        """ 
        第一条错误记录纠正为前一次交易记录的股票余额加上当日净买入，若不是第一条错误记录且
        股票余额等于前一条错误记录的股票余额则股票余额不变，若不是第一条错误记录且股票余额
        不等于前一条错误记录的股票余额则股票余额等于，前一次交易记录的股票余额加上当日净买入。 
        """
        first_ambulance_df['securityBalance'] = \
            first_ambulance_df['extra_number']+first_ambulance_df['securityBalanceLast']
        first_ambulance_df = first_ambulance_df[[
                'fundsAccount','deliveryDate','securitiesCode','securityBalance']].\
                rename(columns={'securityBalance':'securityBalanceUpdate'})
        data = data.merge(first_ambulance_df,
                          on=['fundsAccount','deliveryDate','securitiesCode'],how='left')
        data['securityBalanceLastUpdate'] = \
            data.groupby(['fundsAccount','securitiesCode'])['securityBalance'].shift(1)
        data['securityBalance'] = np.where(
                data['securityBalanceUpdate'].notnull(), 
                data['securityBalanceUpdate'], 
                np.where(data['securityBalanceLastUpdate']!=data['securityBalanceLast'],
                         data['extra_number']+data['securityBalanceLast'],
                         data['securityBalance']))
        data = data.sort_values(['fundsAccount','deliveryDate','securitiesCode'])
        data.drop(['securityBalanceLastUpdate','securityBalanceUpdate'],axis=1,inplace=True)
        return data
        
    def stock_first_position(self, data):
        data = data.copy()
        data['securityBalance'] = data['securityFirstPosition']
        data.drop(['cash','securityBalanceLast', 'securityFirstPosition'],axis=1,inplace=True)
        
        data = data.merge(self.cash_df, on=['fundsAccount','deliveryDate'], how='left')
        """ 时间初始化为数据最早日期的前一天 """        
        timestamp = data['deliveryDate'].min()-timedelta(days=1)
        data['deliveryDate'] = timestamp
        """ 买入卖出量与金额初始化为0 """       
        data.iloc[:, 4:10] = 0
        return data    