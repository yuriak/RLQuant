# -*- coding:utf-8 -*-
import os
import pandas as pd
import talib
import numpy as np
from collections import OrderedDict
from utils.HuobiServices import *

lmap = lambda func, it: list(map(lambda x: func(x), it))
lfilter = lambda func, it: list(filter(lambda x: func(x), it))


class CryptoCurrencyEnv(object):
    def __init__(self, instruments,
                 access_key,
                 secret_key,
                 base_currency='btc',
                 capital_base=1,
                 data_local_path='./data',
                 re_download=False,
                 commission_fee=5e-3,
                 normalize_length=10,
                 data_interval='60min'
                 ):

        self.secret_key = secret_key
        self.access_key = access_key
        init_account(self.access_key, self.secret_key)

        self.instruments = instruments
        self.base_currency = base_currency
        self.capital_base = capital_base
        self.commission_fee = commission_fee
        self.normalize_length = normalize_length
        self.data_local_path = data_local_path
        self.data_interval = data_interval

        self.market_data = self._init_market_data(re_download=re_download)
        self.pointer = normalize_length - 1
        self.done = (self.pointer == (self.market_data.shape[1] - 1))

        self.current_position = np.zeros(len(self.instruments))
        self.current_portfolio_value = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))
        self.current_weight = np.concatenate((np.zeros(len(self.instruments)), [1.]))
        self.current_date = self.market_data.major_axis[self.pointer]

        self.portfolio_values = []
        self.positions = []
        self.weights = []
        self.trade_dates = []

    def reset(self):
        self.pointer = self.normalize_length
        self.current_position = np.zeros(len(self.instruments))
        self.current_portfolio_value = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))
        self.current_weight = np.concatenate((np.zeros(len(self.instruments)), [1.]))
        self.current_date = self.market_data.major_axis[self.pointer]
        self.done = (self.pointer == (self.market_data.shape[1] - 1))

        self.portfolio_values = []
        self.positions = []
        self.weights = []
        self.trade_dates = []

        return self._get_normalized_state(), self.done

    def step(self, action):
        assert action.shape[0] == len(self.instruments) + 1
        assert np.sum(action) <= 1 + 1e5
        current_price = self.market_data[:, :, 'close'].iloc[self.pointer].values
        self._rebalance(action=action, current_price=current_price)

        self.pointer += 1
        self.done = (self.pointer == (self.market_data.shape[1] - 1))
        next_price = self.market_data[:, :, 'close'].iloc[self.pointer].values
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
        self.current_date = self.market_data.major_axis[self.pointer]

        self.positions.append(self.current_position.copy())
        self.weights.append(self.current_weight.copy())
        self.portfolio_values.append(self.current_portfolio_value.copy())
        self.trade_dates.append(self.current_date)

    def _get_normalized_state(self):
        data = self.market_data.iloc[:, self.pointer + 1 - self.normalize_length:self.pointer + 1, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
        return state

    def get_meta_state(self):
        return self.market_data.iloc[:, self.pointer, :]

    def _get_reward(self, current_price, next_price):
        return_rate = (next_price / current_price)
        log_return = np.log(return_rate)
        last_weight = self.current_weight.copy()
        securities_value = self.current_portfolio_value[:-1] * return_rate
        self.current_portfolio_value[:-1] = securities_value
        self.current_weight = self.current_portfolio_value / np.sum(self.current_portfolio_value)
        reward = last_weight[:-1] * log_return
        return reward

    def _init_market_data(self, data_name='crc_market_data.pkl', re_download=False):
        data_path = self.data_local_path + '/' + data_name
        if not os.path.exists(self.data_local_path):
            os.mkdir(self.data_local_path)
        if not os.path.exists(data_path) or re_download:
            print('Start to download crc market data')
            market_data = CryptoCurrencyEnv.klines(instruments=self.instruments,
                                                   base_currency=self.base_currency,
                                                   interval=self.data_interval)
            market_data = CryptoCurrencyEnv._pre_process(market_data, open_c='open', high_c='high', low_c='low', close_c='close', volume_c='vol')
            market_data.to_pickle(data_path)
            print('Done')
        else:
            print('market data exist, loading')
            market_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
        return market_data

    def get_summary(self):
        portfolio_value_df = pd.DataFrame(np.array(self.portfolio_values), index=np.array(self.trade_dates), columns=self.instruments + ['cash'])
        positions_df = pd.DataFrame(np.array(self.positions), index=np.array(self.trade_dates), columns=self.instruments)
        weights_df = pd.DataFrame(np.array(self.weights), index=np.array(self.trade_dates), columns=self.instruments + ['cash'])
        return portfolio_value_df, positions_df, weights_df

    @staticmethod
    def _pre_process(market_data, open_c, high_c, low_c, close_c, volume_c):
        market_data = lmap(lambda x: (x[0], CryptoCurrencyEnv._get_indicators(x[1], close_name=close_c, high_name=high_c, low_name=low_c, open_name=open_c, volume_name=volume_c)), market_data)
        market_data = OrderedDict(market_data)
        market_data = pd.Panel(market_data)
        return market_data

    @staticmethod
    def kline(instrument, base_currency='btc', interval='60min', count=2000):
        s = get_kline('{0}{1}'.format(instrument, base_currency), interval, count)
        if s is None: return None
        s = s['data']
        s = pd.DataFrame(s)[::-1]
        if s.shape[0] < count:
            return None
        s.index = pd.DatetimeIndex(s['id'].apply(lambda x: datetime.datetime.utcfromtimestamp(x) + datetime.timedelta(hours=8)))
        s = s.drop('id', axis=1)
        s['AVG'] = (np.mean(s[['open', 'high', 'low', 'close']], axis=1))
        s['LOG_RR'] = np.log(s['close'] / s['close'].shift(1)).fillna(0)
        s['RR'] = s['close'] / s['close'].shift(1)
        return s

    @staticmethod
    def klines(instruments, base_currency='btc', interval='60min', count=2000):
        return lfilter(lambda x: x[1] is not None, lmap(lambda x: (x, CryptoCurrencyEnv.kline(x, base_currency=base_currency, interval=interval, count=count)), instruments))

    @staticmethod
    def _get_indicators(stock, open_name, close_name, high_name, low_name, volume_name='vol'):
        open_price = stock[open_name].values
        close_price = stock[close_name].values
        low_price = stock[low_name].values
        high_price = stock[high_name].values
        volume = stock[volume_name].values
        data = stock.copy()
        data['MOM'] = talib.MOM(close_price)
        data['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
        data['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
        data['sine'], data['leadsine'] = talib.HT_SINE(close_price)
        data['inphase'], data['quadrature'] = talib.HT_PHASOR(close_price)
        data['ADXR'] = talib.ADXR(high_price, low_price, close_price)
        data['APO'] = talib.APO(close_price)
        data['AROON_UP'], _ = talib.AROON(high_price, low_price)
        data['CCI'] = talib.CCI(high_price, low_price, close_price)
        data['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price)
        data['PPO'] = talib.PPO(close_price)
        data['macd'], data['macd_sig'], data['macd_hist'] = talib.MACD(close_price)
        data['CMO'] = talib.CMO(close_price)
        data['ROCP'] = talib.ROCP(close_price)
        data['fastk'], data['fastd'] = talib.STOCHF(high_price, low_price, close_price)
        data['TRIX'] = talib.TRIX(close_price)
        data['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price)
        data['WILLR'] = talib.WILLR(high_price, low_price, close_price)
        data['NATR'] = talib.NATR(high_price, low_price, close_price)
        data['MFI'] = talib.MFI(high_price, low_price, close_price, volume)
        data['RSI'] = talib.RSI(close_price)
        data['AD'] = talib.AD(high_price, low_price, close_price, volume)
        data['OBV'] = talib.OBV(close_price, volume)
        data['EMA'] = talib.EMA(close_price)
        data['SAREXT'] = talib.SAREXT(high_price, low_price)
        data['TEMA'] = talib.EMA(close_price)
        data = data.drop([open_name, high_name, low_name, 'amount', 'count'], axis=1)
        data = data.dropna().astype(np.float32)
        return data
