# -*- coding:utf-8 -*-
import pandas as pd
import zipline
from DRL_Portfolio import DRL_Portfolio
import logbook
import talib
import sys
import requests
import itertools
import sys
import os

from zipline.api import order_target, record, symbol, order_target_percent, set_benchmark, order_target
from zipline.finance import commission, slippage
import matplotlib.pyplot as plt
import numpy as np

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(level=logbook.DEBUG),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()
from ZiplineTensorboard import TensorBoard


def generate_tech_data(stock):
    price = stock.values
    name = stock.name
    data = pd.DataFrame(stock)
    data[name + '_mom'] = talib.MOM(price)
    data[name + '_macd'], data[name + '_macd_sig'], data[name + '_macd_hist'] = talib.MACD(price)
    data[name + '_rsi'] = talib.RSI(price, timeperiod=10)
    data[name + '_cmo'] = talib.CMO(price)
    data = data.drop(name, axis=1)
    data = data.dropna()
    return data


z_score = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)


def batch_nomorlize(f_data):
    need_normalize = f_data.columns[list(f_data.columns.map(lambda x: '_' in x))]
    keep_original = f_data.columns[list(f_data.columns.map(lambda x: '_' not in x))]
    return z_score(f_data[need_normalize]).join(f_data[keep_original])

def initialize(context):
    #     set_benchmark(symbol('SPY'))
    model.init_model()
    context.i = 0
    context.assets = list(map(lambda x:symbol(x), high_cap_company.Symbol.values))
    context.model_fee = 1e-3
    context.previous_predict_reward = 0
    context.previous_action = 0
    context.set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    context.sequence_length = 300
    context.tb_log_dir = './log/backtest'
    context.tensorboard = TensorBoard(log_dir=context.tb_log_dir)


def before_trading_start(context, data):
    assets_history = data.history(context.assets, ['price','volume'], bar_count=context.sequence_length, frequency='1d')[:-1]
    symbols=assets_history['price'].columns.map(lambda x: x.symbol)
    prices= assets_history['price']
    volumes=np.log(assets_history['volume'])
    full_features=pd.concat(tuple([generate_tech_data(prices[c]) for c in symbols]), axis=1)
    return_rate=(prices/prices.shift(1))[full_features.index[0]:]
    log_return_rate=np.log(return_rate)
    f_data = full_features.join(log_return_rate)
    z_data = return_rate.join(pd.Series(np.ones((f_data.shape[0])) * 1.0001, index=f_data.index, name='ASSET'))[f_data.index[0]:]
    hidden_initial_state, current_rnn_output = model.get_rnn_zero_state()
    feed = model.build_feed_dict(batch_F=batch_nomorlize(f_data),
                                 batch_Z=z_data,
                                 keep_prob=0.8,
                                 fee=context.model_fee,
                                 rnn_hidden_init_state=hidden_initial_state,
                                 output_hidden_init_state=current_rnn_output,
                                 initial_output=current_rnn_output)
    rewards, cum_reward, actions, hidden_initial_state, output_initial_state, current_rnn_output = model.trade(feed)
    while cum_reward < 0.5:
        model.train(feed=feed)
        rewards, cum_reward, actions, current_state, current_rnn_output = model.trade(feed)
    context.today_action = actions[-1].flatten()

def handle_data(context, data):
    # trading_date = data.history(context.assets, ['close'], bar_count=1, frequency='1d').index[0].date()
    trading_date=context.get_datetime().date
    context.i += 1
    record(action=context.today_action)
    action = context.today_action.flatten()[:-1]
    for k,asset in enumerate(context.assets):
        order_target_percent(asset, action[k])
    if context.i % 100 == 0:
        print(trading_date)
        print(context.portfolio.portfolio_value)
    if context.tensorboard is not None:
        # record algo stats to tensorboard
        context.tensorboard.log_algo(context, epoch=context.i)


if __name__ == '__main__':
    if not os.path.exists('sp500.csv'):
        print('downloading sp500 data')
        with open('sp500.csv', 'wb+') as f:
            response = requests.get('https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
            f.write(response.content)
    sp500 = pd.read_csv('sp500.csv')
    sp500.index = sp500['Symbol']
    high_cap_company = sp500.loc[list(itertools.chain.from_iterable(list(map(lambda x: x[1][-3:], list(sp500.sort_values('Market Cap').groupby('Sector').groups.items())))))]
    model = DRL_Portfolio(feature_number=len(high_cap_company)*7,asset_number=len(high_cap_company)+1)
    
    start = pd.Timestamp(pd.to_datetime('2004-02-08')).tz_localize('US/Eastern')
    end = pd.Timestamp(pd.to_datetime('2018-03-27')).tz_localize('US/Eastern')
    result = zipline.run_algorithm(start=start, end=end,
                                   initialize=initialize,
                                   before_trading_start=before_trading_start,
                                   handle_data=handle_data,
                                   capital_base=100000,
                                   data_frequency='daily',
                                   bundle='quandl'
                                   )
    result.to_pickle('portfolio_backtest_result')