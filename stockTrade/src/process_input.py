# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:23:27 2018

@author: chenyi
"""

import pandas as pd

class JzStockData(object):
    
    def __init__(self):
        super(JzStockData, self).__init__()
        self.float_to_str = lambda x: str(int(x)).strip() if type(x) is float else str(x).strip()
        self.nan_to_str = lambda x: str(x).strip() if len(x) > 0 else None
    
    def data_input(self):
        daily_index = [0, 1, 19, 22, 3, 4, 7, 12, 13, 11, 16, 17, 2, 5, 20]
        asset_index = [5, 6, 2, 37, 20, 22, 15, 36, 32, 24, 16, 17, 18, 21, 0]
        daily_cols = [
                'customerCode','fundsAccount','deliveryDate','transactionTime','securitiesCode',
                'securitiesName','abstractCode','transactionPrice','turnoverNumber',
                'securityBalance','fundsNumber','fundBalance','market','securitiesCategory',
                'dateAppointment']
        asset_cols = [
                'custid','fundid','bizdate','matchtime','stkcode','stkname','digestid','matchprice',
                'matchqty','stkbal','fundeffect','fundbal','market','stktype','operdate']
        jz_daily1 = pd.read_table('data/jz_stock_flow_daily.txt', sep=' ', 
                                  usecols = sorted(daily_index), na_filter=None, 
                                  parse_dates=[12], dtype = {'securitiesCode':'str'},
                                  encoding='utf-8', skip_blank_lines = True).\
                                  reindex(columns = daily_cols)
        jz_daily2 = pd.read_table('data/jz_stock_flow_daily_add.txt', sep=' ', 
                                  usecols = sorted(daily_index), na_filter=None,
                                  dtype = {'securitiesCode':'str'}, parse_dates=[12], 
                                  skip_blank_lines=True, encoding='utf-8').\
                                  reindex(columns = daily_cols)
        jz_asset = pd.read_table('data/jz_asset.txt', sep=' ',  
                                 usecols = sorted(asset_index), na_filter=None,
                                 parse_dates=[1], dtype = {'stkcode':'str'},
                                 skip_blank_lines=True, encoding='utf-8').\
                                 rename(columns=dict(zip(asset_cols,daily_cols))).\
                                 reindex(columns=daily_cols)
        return {'jz_daily1':jz_daily1,
                'jz_daily2':jz_daily2,
                'jz_asset':jz_asset}
        
    def transform(self):
        data = self.data_input()
        """ dateAppointment 日期格式转整形 """
        data['jz_asset'].dateAppointment = \
                data['jz_asset'].dateAppointment.str.replace('/', '').astype('int64')
        
        """ Market 转化统一格式 """
        asset_market = ['0', '1', '2', '3', '5', '6', '7', 'S', 'J', 'H']
        daily_market = ['深A','沪A','深B','沪B','沪港通','股转A','股转A','深港通','基金','股转B']
        daily_char = ['深Ａ','沪Ａ','深Ｂ','沪Ｂ','沪港通','股转Ａ','股转Ａ','深港通','基金','股转Ｂ']
        data['jz_daily1']['market'] = data['jz_daily1']['market'].apply(self.market_replace).\
                replace(dict(zip(daily_char, daily_market)))
        data['jz_daily2']['market'] = data['jz_daily2']['market'].apply(self.nan_to_str).\
                replace(dict(zip(daily_char, daily_market)))
        data['jz_asset']['market'] = data['jz_asset']['market'].apply(self.float_to_str).\
                apply(self.nan_to_str).replace(dict(zip(asset_market, daily_market)))
                
        """ SecuritiesCategory 转化统一格式 """
        daily_sec_category = [
                '股票','挂牌公司证券','B转H股','公司2债','国债','公司债','国债','投资基金','债券转股',
                '公司债','转让资管计划','理财产品','可交换公司债','上证LOF','ETF','货币ETF','指定交易',
                '报价回购','LOF','质押回购','黄金ETF','议案投票']
        asset_sec_category = ['0','J','f','2','g','Q','1','o','9','C','m',
                              '5','8','r','E','h','Z','K','L','G','i','T'] 
        data['jz_daily1']['securitiesCategory'] = \
                data['jz_daily1']['securitiesCategory'].apply(self.nan_to_str)
        data['jz_daily2']['securitiesCategory'] = \
                data['jz_daily2']['securitiesCategory'].apply(self.nan_to_str)
        data['jz_asset']['securitiesCategory'] = data['jz_asset']['securitiesCategory'].\
                apply(self.float_to_str).apply(self.nan_to_str).\
                replace(dict(zip(asset_sec_category, daily_sec_category)))
                
        """ 按照时间戳连接数据 """
        asset_range = (data['jz_asset']['deliveryDate'] < '2015-03-22') | \
                        (data['jz_asset']['deliveryDate'] >= '2015-04-09') & \
                        (data['jz_asset']['deliveryDate'] <= '2017-05-24')
        daily1_range = (data['jz_daily1']['deliveryDate'] >= '2017-06-02')
        data = pd.concat([data['jz_daily1'][daily1_range], 
                          data['jz_daily2'], data['jz_asset'][asset_range]], axis=0).\
            sort_values(['fundsAccount','deliveryDate']).reset_index(drop = True)  
        data['securitiesName'] = data['securitiesName'].str.replace(' ','').\
                    apply(self.nan_to_str)
        return data
    
    def market_replace(self, X):
        X = str(X)
        if X.startswith(('A', '0', 'B', '2', 'C')):
            return '沪深'
        elif len(X) == 0:
            return None
        elif (X.endswith('三方') | (X.startswith('建行'))):
            return '银证'
        elif X.startswith(('F', '5', '6', '98', '1')):
            return '基金'
        else:
            return X

    def jz_stock_generate(self, data):
        """ 市场范围 """
        market_category = ['沪A','深A','股转A','股转Ｂ','深B',
                           '沪B','沪港通','深港通','沪深','银证']
        """ 证券范围 """
        security_category = ['股票','债券转股','挂牌公司证券','创业板']
        
        """ 股票代码范围（9开头沪B，200开头深B，40，43，8股转，新三板，其他A股，豫能控股，宗申动力 """
        code_category = ['9','8','60','43','40','300','200',
                         '19','002','000','001696','001896']
        
        """ 股票名称范围 """
        name_category = ['新代码','华夏300','华夏复兴','南方中票C','华夏全球','国富货币A',
                         '南方医保','核心成长','南方新优享','中邮竞争力','中金纯债Ａ',
                         '华夏回二','海盈货币A','国泰货币基金','国开货币A','东方金账簿',
                         '东方龙基金','长城增利C','长城货币B','长城增利A','华富诚鑫C',
                         '国开货币A','南方稳利C','中邮双动力','南方通利A','中国梦基金',
                         '鑫元货币A','工银纯债A','中加货币A','鹏华国企债']
        
        cond_secCode_6 = data['securitiesCode'].apply(
                lambda x: (len(str(x)) == 6) & str(x).startswith(tuple(code_category)))
        cond_secName_6 = data['securitiesName'].apply(
                lambda x: (~(x in name_category)&(len(str(x).strip())<6|('ST' in x))) if x is not None else 1)
        
        """ 股票帐号为空则股票名称不是空，且在市场和证券范围 """
        cond_0 = (data['securitiesCode'].str.len() == 0) &\
                    ~(data.securitiesName.isin([None,'新代码'])) & \
                    data['market'].isin(market_category) & \
                    data['securitiesCategory'].isin(security_category)
        
        """ 股票帐号长度为5且在市场和证券范围 """
        cond_5 = (data['securitiesCode'].str.len() == 5) &\
                    data['market'].isin(market_category) & \
                    data['securitiesCategory'].isin(security_category)
       
        """ 股票帐号长度为6且名词不在基金范围(name_caterory) """
        cond_6 = cond_secCode_6 & cond_secName_6
        
        """ 新代码导入平衡数据 """
        cond_new = (data['securitiesName']=='新代码')&\
                    (data['securitiesCategory']=='股票')&\
                    (data['securitiesCode'].str.startswith(tuple(code_category)))
        data = data[cond_0 | cond_5 | cond_6 | cond_new].\
                sort_values(['customerCode','fundsAccount','deliveryDate', 
                             'securitiesCode', 'transactionTime']).\
                reset_index(drop=True)
        return data