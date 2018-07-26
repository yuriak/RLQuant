# -*- coding:utf-8 -*-
import sys

import logbook
import numpy as np
import pandas as pd
import talib
import zipline
from DRL_PairsTrading import DRL_PairsTrading
from zipline.api import record, symbol, order_target_percent
from zipline.finance import commission, slippage

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(level=logbook.DEBUG),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()
from history.ZiplineTensorboard import TensorBoard

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
    context.previous_action=0
    context.set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    context.sequence_length=300
    context.tb_log_dir='./log/backtest'
    context.tensorboard=TensorBoard(log_dir=context.tb_log_dir)

def before_trading_start(context, data):
    asset1_history = data.history(context.asset1, ['price'], bar_count=context.sequence_length, frequency='1d')[:-1]
    asset2_history = data.history(context.asset2, ['price'], bar_count=context.sequence_length, frequency='1d')[:-1]
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
    rewards, cum_reward, actions, current_state, current_rnn_output = model.trade(feed)
    while cum_reward < 0.5:
        model.train(feed=feed)
        rewards, cum_reward, actions, current_state, current_rnn_output=model.trade(feed)
    # while cum_reward < context.previous_predict_reward:
    #     model_archive.train(feed)
    #     rewards, cum_reward, actions, current_state, current_rnn_output = model_archive.trade(feed)
    #     print('traning: ', cum_reward)
    # context.previous_predict_reward=cum_reward
    context.today_action=actions[-1].ravel()[0]


def my_round(x):
    if x < -0.95:
        return -1
    elif x > 0.95:
        return 1
    else:
        return x
    
def handle_data(context, data):
    trading_date= data.history(context.asset1, ['close'], bar_count=1, frequency='1d').index[0].date()
    context.i += 1
    asset1_close = data.history(context.asset1, ['price'], bar_count=1, frequency='1d').price.values[0]
    asset2_close = data.history(context.asset2, ['price'], bar_count=1, frequency='1d').price.values[0]
    spread = asset1_close - asset2_close
    record(S1=asset1_close,
           S2=asset2_close,
           spread=spread,
           action=context.today_action)
    action=my_round(context.today_action)
    if action != context.previous_action:
        signal_percent = action / 2
        order_target_percent(context.asset1, signal_percent)
        order_target_percent(context.asset2, -signal_percent)
    context.previous_action=action
    if context.i % 100 == 0:
        print(trading_date)
        print(context.portfolio)
    if context.tensorboard is not None:
        # record algo stats to tensorboard
        context.tensorboard.log_algo(context,epoch=context.i)

if __name__ == '__main__':
    start = pd.Timestamp(pd.to_datetime('2002-02-08')).tz_localize('US/Eastern')
    end = pd.Timestamp(pd.to_datetime('2018-03-27')).tz_localize('US/Eastern')
    pairsTrading= zipline.algorithm.TradingAlgorithm()
    result = zipline.run_algorithm(start=start, end=end,
                                   initialize=initialize,
                                   before_trading_start=before_trading_start,
                                   handle_data=handle_data,
                                   capital_base=100000,
                                   data_frequency='daily',
                                   bundle='quandl'
                                   )
    result.to_pickle('backtest_result')