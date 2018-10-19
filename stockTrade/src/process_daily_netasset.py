# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:49:49 2018

@author: chenyi
"""
from process_stock_detection import JzStockDetection
class JzNetAsset(JzStockDetection):
    
    def __init__(self):
        super(JzStockDetection,self).__init__()
    
    def jz_netasset_generate(self, data, jz_data):
        data = data.copy()
        transfer_df = self.daily_transfer_inout(jz_data)
        data = data.merge(transfer_df, on=[
                'fundsAccount','deliveryDate'],how='left').fillna(0)

        data_assetfirst = data[['fundsAccount','deliveryDate','totalAsset']].\
                groupby(['fundsAccount']).first().\
                reset_index().rename(columns={'totalAsset':'netAsset'})
                
        data = data.merge(data_assetfirst, on=[
                'fundsAccount','deliveryDate'], how='left').fillna(0)
        
        data['netAsset'] = data['netAsset']+data['TransferIn']+data['TransferOut']
#        data = data.groupby(['fundsAccount'],as_index=False).\
#                apply(self.AddnetAsset).reset_index(drop=True)
        data['netAsset'] = data.groupby(['fundsAccount'])['netAsset'].cumsum().values
        data['assetRatio'] = data['totalAsset']/data['netAsset']
        return data

    def AddnetAsset(self, data):
        df = data.copy()
        df.netasset = df.netasset.cumsum()
        return df
        