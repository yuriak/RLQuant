# -*- coding:utf-8 -*-
import os
import quandl
import pandas as pd
import talib
import numpy as np
from utils.DataUtils import generate_tech_data_default


class Market(object):
    def __init__(self, instruments, api_key, start_date='2002-01-01', end_date=None, local_path='./data'):
        super(Market, self).__init__()
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.local_path = local_path
        quandl.ApiConfig.api_key = self.api_key
        self.market_data = self._init_market_data()

    def _init_market_data(self, data_name='market_data.pkl', pre_process=True):
        data_path = self.local_path + '/' + data_name
        if not os.path.exists(data_path):
            print('Start to download market data')
            market_data = {}
            for s in self.instruments:
                print('downloading', s)
                stock = quandl.get_table('EOD/{0}'.format(s), start_date=self.start_date, end_date=self.end_date)
                market_data[s] = stock
            market_data = pd.Panel(market_data).fillna(method='ffill').fillna(method='bfill')
            market_data.to_pickle(data_path)
            print('Done')
        else:
            print('market data exist, loading')
            market_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
        if pre_process:
            market_data = Market.pre_process(market_data, open_c='Adj_Open', close_c='Adj_Close', high_c='Adj_High', low_c='Adj_Low')
        assert np.sum(np.isnan(market_data.values)) == 0
        return market_data

    @staticmethod
    def pre_process(market_data, open_c, high_c, low_c, close_c, volume_c):
        preprocessed_data = {}
        for c in market_data.items:
            columns = [open_c, close_c, high_c, low_c, volume_c]
            security = market_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
            security[volume_c] = security[volume_c].replace(0, np.nan).fillna(method='ffill')
            tech_data = Market.get_indicators(security=security.astype(float), close_name=close_c, high_name=high_c, low_name=low_c, volume_name=volume_c)
            preprocessed_data[c] = tech_data
        return pd.Panel(preprocessed_data).dropna()

    @staticmethod
    def get_indicators(security, close_name, high_name, low_name, volume_name):
        close_price = security[close_name].values
        low_price = security[low_name].values
        high_price = security[high_name].values
        volume = security[volume_name].values if volume_name else None
        security['MOM'] = talib.MOM(close_price)
        security['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
        security['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
        security['SINE'], security['LEADSINE'] = talib.HT_SINE(close_price)
        security['INPHASE'], security['QUADRATURE'] = talib.HT_PHASOR(close_price)
        security['ADXR'] = talib.ADXR(high_price, low_price, close_price)
        security['APO'] = talib.APO(close_price)
        security['AROON_UP'], _ = talib.AROON(high_price, low_price)
        security['CCI'] = talib.CCI(high_price, low_price, close_price)
        security['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price)
        security['PPO'] = talib.PPO(close_price)
        security['MACD'], security['MACD_SIG'], security['MACD_HIST'] = talib.MACD(close_price)
        security['CMO'] = talib.CMO(close_price)
        security['ROCP'] = talib.ROCP(close_price)
        security['FASTK'], security['FASTD'] = talib.STOCHF(high_price, low_price, close_price)
        security['TRIX'] = talib.TRIX(close_price)
        security['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price)
        security['WILLR'] = talib.WILLR(high_price, low_price, close_price)
        security['NATR'] = talib.NATR(high_price, low_price, close_price)
        security['RSI'] = talib.RSI(close_price)
        security['EMA'] = talib.EMA(close_price)
        security['SAREXT'] = talib.SAREXT(high_price, low_price)
        security['TEMA'] = talib.EMA(close_price)
        security['RR'] = security[close_name] / security[close_name].shift(1).fillna(1)
        security['LOG_RR'] = np.log(security['RR'])
        if volume:
            security['MFI'] = talib.MFI(high_price, low_price, close_price, volume)
            security['AD'] = talib.AD(high_price, low_price, close_price, volume)
            security['OBV'] = talib.OBV(close_price, volume)
            security[volume_name] = np.log(security[volume_name])
        security = security.dropna().astype(np.float32)
        return security
