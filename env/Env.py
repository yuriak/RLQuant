# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import quandl
import os


class ENV(object):
    def __init__(self, instruments, capital_base=1e5, start_date='2002-01-01', end_date=None):
        self.instruments = instruments
        self.capital_base = capital_base
        self.start_date = start_date
        self.end_date = end_date
        self.market_data = self._init_market_data()
        self.baseline = None
        self.portfolio = None
        
        pass
    
    def _init_market_data(self, pre_process=True, data_path='./data/market_data'):
        if not os.path.exists(data_path):
            print('Start to download good history data')
            market_data = {}
            for s in self.instruments:
                print('downloading', s)
                stock = quandl.get_table('WIKI/PRICES', date={'gte': str(start_date)}, ticker=s)
                stock.index = stock.date
                market_data[s] = stock
            market_data = pd.Panel(market_data).fillna(method='ffill').fillna(method='bfill')
            market_data.to_pickle(data_path)
            market_data = generate_stock_features(market_data, max_time_window=max_time_window)
            print('Done')
        else:
            print('equity data exist')
            market_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
            market_data = generate_stock_features(market_data, max_time_window=max_time_window)
        assert np.sum(np.isnan(market_data.values)) == 0
        return market_data
    
    def reset(self):
        pass
    
    def step(self):
        pass
