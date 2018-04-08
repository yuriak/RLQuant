# -*- coding:utf-8 -*-
import pandas as pd
import zipline
from DRL_PairsTrading import DRL_PairsTrading
import talib

from zipline.api import order_target, record, symbol, order_target_percent, set_benchmark, order_target
import matplotlib.pyplot as plt
import numpy as np



def generate_tech_data(p1_df,p2_df):
    sample = pd.DataFrame({'p1': p1_df.values.ravel(), 'p2': p2_df.values.ravel()}, index=p1_df.index)
    p1=p1_df.values.ravel()
    p2=p2_df.values.ravel()
    sample['p1'+'_mom'] = talib.MOM(p1)
    sample['p1' + '_macd'], sample['p1' + '_macd_sig'], sample['p1' + '_macd_hist'] = talib.MACD(p1)
    sample['p1' + '_rsi'] = talib.RSI(p1, timeperiod=10)
    sample['p1' + '_cmo'] = talib.CMO(p1)
    sample['p2' + '_mom'] = talib.MOM(p2)
    sample['p2' + '_macd'], sample['p2' + '_macd_sig'], sample['p2' + '_macd_hist'] = talib.MACD(p2)
    sample['p2' + '_rsi'] = talib.RSI(p2, timeperiod=10)
    sample['p2' + '_cmo'] = talib.CMO(p2)
    spread = p1 / p2
    sample['spread'] = spread
    sample['diff'] = sample['spread'] / sample['spread'].shift(1)
    sample = sample.dropna()
    return sample


z_score = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)

def batch_nomorlize(sample):
    
    return z_score(sample)

model=DRL_PairsTrading(16)

def initialize(context):
    #     set_benchmark(symbol('SPY'))
    model.init_model()
    context.i = 0
    context.asset1 = symbol('EBAY')
    context.asset2 = symbol('KLAC')
    context.model_fee= 1e-3
    context.previous_predict_reward=0

def before_trading_start(context, data):
    asset1_history = data.history(context.asset1, ['open'], bar_count=1000, frequency='1d')
    asset2_history = data.history(context.asset2, ['open'], bar_count=1000, frequency='1d')
    samples = generate_tech_data(asset1_history, asset2_history)
    context.trading_date=asset1_history.index[-1].date()
    f_data = samples
    z_data = np.expand_dims(samples['diff'], axis=1)
    hidden_initial_state = model.get_rnn_zero_state()
    current_rnn_output = hidden_initial_state[-1]
    feed = model.build_feed_dict(batch_F=batch_nomorlize(f_data),
                                 batch_Z=z_data,
                                 keep_prob=0.8,
                                 fee=context.model_fee,
                                 rnn_hidden_init_state=hidden_initial_state,
                                 previous_output=current_rnn_output)
    model.train(feed=feed)
    model.train(feed=feed)
    rewards, cum_reward, actions, current_state, current_rnn_output=model.trade(feed)
    # while cum_reward < context.previous_predict_reward:
    #     model.train(feed)
    #     rewards, cum_reward, actions, current_state, current_rnn_output = model.trade(feed)
    #     print('traning: ', cum_reward)
    # context.previous_predict_reward=cum_reward
    context.today_action=actions[-1].ravel()[0]

def handle_data(context, data):
    trading_date=data.current
    context.i += 1
    asset1_close = data.current(context.asset1, 'close')
    asset2_close = data.current(context.asset2, 'close')
    spread = asset1_close - asset2_close
    record(S1=data.current(context.asset1, 'close'),
           S2=data.current(context.asset2, 'close'),
           spread=spread,
           action=context.today_action
           )
    action=context.today_action
    signal_persent = action / 2
    print(signal_persent)
    order_target_percent(context.asset1, signal_persent)
    order_target_percent(context.asset2, -signal_persent)
    print(context.trading_date)

if __name__ == '__main__':
    start = pd.Timestamp(pd.to_datetime('2002-02-08')).tz_localize('US/Eastern')
    end = pd.Timestamp(pd.to_datetime('2018-03-27')).tz_localize('US/Eastern')
    result = zipline.run_algorithm(start=start, end=end,
                                   initialize=initialize,
                                   before_trading_start=before_trading_start,
                                   handle_data=handle_data,
                                   capital_base=100000,
                                   data_frequency='daily',
                                   bundle='quandl'
                                   )
    result.to_pickle('backtest_result')