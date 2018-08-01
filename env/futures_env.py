# -*- coding:utf-8 -*-
import quandl
import pandas as pd
import talib
import numpy as np
import os


class FuturesEnv(object):
    def __init__(self, instruments,
                 api_key,
                 capital_base=1e5,
                 start_date='2002-01-01',
                 end_date=None,
                 data_local_path='./data',
                 re_download=False,
                 commission_fee=5e-3,
                 normalize_length=10
                 ):
        self.api_key = api_key
        quandl.ApiConfig.api_key = self.api_key
        self.instruments = instruments
        self.capital_base = capital_base
        self.commission_fee = commission_fee
        self.normalize_length = normalize_length
        self.start_date = start_date
        self.end_date = end_date
        self.data_local_path = data_local_path
        self.preprocessed_market_data, self.cleaned_market_data = self._init_market_data(re_download=re_download)
        self.pointer = normalize_length - 1
        self.done = (self.pointer == (self.preprocessed_market_data.shape[1] - 1))
        
        self.current_position = np.zeros(len(self.instruments))
        self.current_portfolio_value = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))
        self.current_weight = np.concatenate((np.zeros(len(self.instruments)), [1.]))
        self.current_date = self.preprocessed_market_data.major_axis[self.pointer]
        
        self.portfolio_values = []
        self.positions = []
        self.weights = []
        self.trade_dates = []
    
    def reset(self):
        self.pointer = self.normalize_length
        self.current_position = np.zeros(len(self.instruments))
        self.current_portfolio_value = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))
        self.current_weight = np.concatenate((np.zeros(len(self.instruments)), [1.]))
        self.current_date = self.preprocessed_market_data.major_axis[self.pointer]
        self.done = (self.pointer == (self.preprocessed_market_data.shape[1] - 1))
        
        self.portfolio_values = []
        self.positions = []
        self.weights = []
        self.trade_dates = []
        
        return self._get_normalized_state(), self.done
    
    def step(self, action):
        assert action.shape[0] == len(self.instruments) + 1
        assert np.sum(action) <= 1 + 1e5
        current_price = self.cleaned_market_data[:, :, 'Last'].iloc[self.pointer].values
        self._rebalance(action=action, current_price=current_price)
        
        self.pointer += 1
        self.done = (self.pointer == (self.preprocessed_market_data.shape[1] - 1))
        next_price = self.cleaned_market_data[:, :, 'Last'].iloc[self.pointer].values
        reward = self._get_reward(current_price=current_price, next_price=next_price)
        state = self._get_normalized_state()
        return state, reward, self.done
    
    def _rebalance(self, action, current_price):
        target_weight = action
        target_value = np.sum(self.current_portfolio_value) * target_weight
        target_position = target_value[:-1] / current_price
        trade_amount = target_position - self.current_position
        commission_cost = np.sum(self.commission_fee * np.abs(trade_amount) * current_price)
        
        self.current_position = target_position
        self.current_portfolio_value = target_value - commission_cost
        self.current_weight = target_weight
        self.current_date = self.preprocessed_market_data.major_axis[self.pointer]
        
        self.positions.append(self.current_position.copy())
        self.weights.append(self.current_weight.copy())
        self.portfolio_values.append(self.current_portfolio_value.copy())
        self.trade_dates.append(self.current_date)
    
    def _get_normalized_state(self):
        data = self.preprocessed_market_data.iloc[:, self.pointer + 1 - self.normalize_length:self.pointer + 1, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
        return state
    
    def get_meta_state(self):
        return self.preprocessed_market_data.iloc[:, self.pointer, :]
    
    def _get_reward(self, current_price, next_price):
        return_rate = (next_price / current_price)
        log_return = np.log(return_rate)
        last_weight = self.current_weight.copy()
        securities_value = self.current_portfolio_value[:-1] * return_rate
        self.current_portfolio_value[:-1] = securities_value
        self.current_weight = self.current_portfolio_value / np.sum(self.current_portfolio_value)
        reward = last_weight[:-1] * log_return
        return reward
    
    def _init_market_data(self, data_name='futures_market_data.pkl', pre_process=True, re_download=False):
        data_path = self.data_local_path + '/' + data_name
        futures = {}
        if not os.path.exists(data_path) or re_download:
            print('Start to download market data')
            for i in self.instruments:
                future = quandl.get('CHRIS/{0}'.format(i), authtoken=self.api_key)
                future = future[['Open', 'High', 'Low', 'Last', 'Volume']]
                futures[i] = future
            market_data = pd.Panel(futures).fillna(method='ffill').fillna(method='bfill')
            market_data.to_pickle(data_path)
            print('Done')
        else:
            print('market data exist, loading')
            market_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
        if pre_process:
            processed_market_data, cleaned_market_data = self._pre_process(market_data, open_c='Open', close_c='Last', high_c='High', low_c='Low', volume_c='Volume')
        assert np.sum(np.isnan(processed_market_data.values)) == 0
        assert np.sum(np.isnan(cleaned_market_data.values)) == 0
        return processed_market_data, cleaned_market_data
    
    def get_summary(self):
        portfolio_value_df = pd.DataFrame(np.array(self.portfolio_values), index=np.array(self.trade_dates), columns=self.instruments + ['cash'])
        positions_df = pd.DataFrame(np.array(self.positions), index=np.array(self.trade_dates), columns=self.instruments)
        weights_df = pd.DataFrame(np.array(self.weights), index=np.array(self.trade_dates), columns=self.instruments + ['cash'])
        return portfolio_value_df, positions_df, weights_df
    
    def _pre_process(self, market_data, open_c, high_c, low_c, close_c, volume_c):
        preprocessed_data = {}
        cleaned_data = {}
        for c in market_data.items:
            columns = [open_c, close_c, high_c, low_c, volume_c]
            security = market_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
            security[volume_c] = security[volume_c].replace(0, np.nan).fillna(method='ffill')
            cleaned_data[c] = security.copy()
            tech_data = FuturesEnv._get_indicators(security=security.astype(float), open_name=open_c, close_name=close_c, high_name=high_c, low_name=low_c, volume_name=volume_c)
            preprocessed_data[c] = tech_data
        preprocessed_data = pd.Panel(preprocessed_data).dropna()
        cleaned_data = pd.Panel(cleaned_data)[:, preprocessed_data.major_axis, :].dropna()
        return preprocessed_data[:, self.start_date:self.end_date, :], cleaned_data[:, self.start_date:self.end_date, :]
    
    @staticmethod
    def _get_indicators(security, open_name, close_name, high_name, low_name, volume_name):
        open_price = security[open_name].values
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
        security['RR'] = security[close_name] / security[close_name].shift(1).fillna(1)
        security['LOG_RR'] = np.log(security['RR'])
        if volume_name:
            security['MFI'] = talib.MFI(high_price, low_price, close_price, volume)
            security[volume_name] = np.log(security[volume_name])
        security.drop([open_name, close_name, high_name, low_name], axis=1)
        security = security.dropna().astype(np.float32)
        return security
