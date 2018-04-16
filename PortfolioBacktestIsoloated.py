# -*- coding:utf-8 -*-
import pandas as pd
import zipline
import quandl
from DRL_Portfolio_Isolated import DRL_Portfolio
import logbook
import talib
import sys
import requests
import itertools
import sys
import os
import tensorflow as tf

from zipline.api import order_target, record, symbol, order_target_percent, set_benchmark, order_target
from zipline.finance import commission, slippage
from zipline.data import bundles
import matplotlib.pyplot as plt
import numpy as np

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(level=logbook.DEBUG),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()
from ZiplineTensorboard import TensorBoard

quandl.ApiConfig.api_key = 'CTq2aKvtCkPPgR4L_NFs'


def generate_tech_data(stock):
    price = stock.values
    name = stock.name
    data = pd.DataFrame(stock)
    data[name + '_mom'] = talib.MOM(price)
    data[name + '_macd'], data[name + '_macd_sig'], data[name + '_macd_hist'] = talib.MACD(price)
    data[name + '_rsi'] = talib.RSI(price, timeperiod=10)
    data[name + '_cmo'] = talib.CMO(price)
    data = data.drop(name, axis=1)
    data = data.fillna(method='bfill')
    return data


z_score = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)


def batch_nomorlize(f_data):
    need_normalize = f_data.columns[list(f_data.columns.map(lambda x: '_' in x))]
    keep_original = f_data.columns[list(f_data.columns.map(lambda x: '_' not in x))]
    return z_score(f_data[need_normalize]).join(f_data[keep_original])


def normallize_all(f_data):
    return z_score(f_data)


def initialize(context):
    context.set_benchmark(None)
    context.i = 1
    context.assets = list(map(lambda x: symbol(x), high_cap_company.Symbol.values))
    print(context.assets, len(context.assets))
    context.model_fee = 1e-2
    context.set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    context.bootstrap_sequence_length = 300
    context.max_sequence_length = 60
    context.tb_log_dir = './log/%s' % back_test_name
    context.model_update_time = 30
    context.target_profit_multiplier = 1.1
    bundle = bundles.load('quandl')
    start_date_str = str(context.get_datetime().date())
    initial_history_start_date = bundle.equity_daily_bar_reader.sessions[bundle.equity_daily_bar_reader.sessions < start_date_str][(-context.bootstrap_sequence_length - 1)]
    initial_history_end_date = bundle.equity_daily_bar_reader.sessions[bundle.equity_daily_bar_reader.sessions > start_date_str][0]
    filterd_assets_index = (np.isnan(np.sum(bundle.equity_daily_bar_reader.load_raw_arrays(columns=['close'], start_date=initial_history_start_date, end_date=initial_history_end_date, assets=context.assets), axis=1)).flatten() == False)
    context.assets = list(np.array(context.assets)[filterd_assets_index])
    print(context.assets, len(context.assets))
    remain_symbols = list(map(lambda x: x.symbol, context.assets))
    if not os.path.exists('history_data'):
        print('Start to download good history data')
        history_data = {}
        for s in remain_symbols:
            print('downloading', s)
            stock = quandl.get_table('WIKI/PRICES', date={'gte': str(initial_history_start_date)}, ticker=s)
            stock.index = stock.date
            history_data[s] = stock
        history_data = pd.Panel(history_data)
        # history_data = history_data.transpose(2, 1, 0)
        history_data.to_pickle('history_data')
        context.history_data = history_data
        print('Done')
    else:
        print('history data exist')
        history_data = pd.read_pickle('history_data')
        context.history_data = history_data
    
    if not os.path.exists('index'):
        print('downloading index data')
        spy = quandl.get("CHRIS/CME_SP1", authtoken="CTq2aKvtCkPPgR4L_NFs")
        gc = quandl.get("CHRIS/CME_GC1", authtoken="CTq2aKvtCkPPgR4L_NFs")
        si = quandl.get("CHRIS/CME_SI1", authtoken="CTq2aKvtCkPPgR4L_NFs")
        vix = pd.read_csv('http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv')
        vix.columns = vix.iloc[0]
        vix = vix[1:]
        vix.index = pd.DatetimeIndex(vix.Date)
        vix = vix.drop('Date', axis=1)
        vix = vix.astype(np.float64)
        vix.columns = ['Open', 'High', 'Low', 'Last']
        index_data = pd.Panel({'vix': vix, 'gc': gc, 'si': si, 'spy': spy})
        # index_data = index_data.transpose(2, 1, 0)
        index_data.to_pickle('index')
        context.index_data = index_data[:, str(initial_history_start_date):, :]
    else:
        print('index data exist')
        index_data = pd.read_pickle('index')
        # index_data = index_data.transpose(2,1,0)
        context.index_data = index_data[:, str(initial_history_start_date):, :]
    
    feature_network_topology = {
        'equity_network': {
            'feature_map_number': len(context.assets),
            'feature_number': 8,
            'input_name': 'equity',
            'dense': {
                'n_units': [256, 512, 256],
                'act': [tf.nn.tanh] * 3,
            },
            'rnn': {
                'n_units': [128, 1],
                'act': [tf.nn.tanh, tf.nn.tanh],
                'attention_length': 10
            },
            'keep_output': True
        },
        'index_network': {
            'feature_map_number': len(context.index_data.items),
            'feature_number': 7,
            'input_name': 'index',
            'dense': {
                'n_units': [32, 64, 32],
                'act': [tf.nn.tanh] * 3,
            },
            'rnn': {
                'n_units': [16, 8],
                'act': [tf.nn.tanh, tf.nn.tanh],
                'attention_length': 10
            },
            'keep_output': False
        }
    }
    context.model = DRL_Portfolio(asset_number=len(context.assets),
                                  feature_network_topology=feature_network_topology,
                                  action_network_layers=[128, 64],
                                  object_function='reward')
    context.real_return = []
    context.model.init_model()
    context.tensorboard = TensorBoard(log_dir=context.tb_log_dir, session=context.model.get_session())


def before_trading_start(context, data):
    trading_date = context.get_datetime().date()
    prices = context.history_data[:, :str(trading_date), 'adj_close'][:-1]
    history_data = context.history_data[:, :str(trading_date), :]
    index_data = context.index_data[:, :str(trading_date), :]
    spy_index = context.index_data['spy', :str(trading_date), 'Last'][:-1].fillna(method='ffill')
    stock_features = {}
    for c in history_data.items:
        stock_data = history_data[c, :-1, ['adj_close', 'adj_volume']].fillna(method='bfill')
        tech_data = generate_tech_data(stock_data['adj_close'].astype(float))
        stock_data['log_volume'] = np.log(stock_data['adj_volume'])
        tech_data = tech_data.join(stock_data['log_volume'])
        log_return = np.log((stock_data['adj_close'] / stock_data['adj_close'].shift(1)).fillna(1))
        tech_data = tech_data.join(log_return)
        tech_data = normallize_all(tech_data[-context.max_sequence_length:])
        stock_features[c] = tech_data
    stock_features = pd.Panel(stock_features)
    index_features = {}
    for i in index_data.items:
        index = index_data[i, :-1, 'Last'].fillna(method='bfill')
        tech_data = generate_tech_data(index)
        log_return = np.log((index / index.shift(1)).fillna(1))
        tech_data = tech_data.join(log_return)
        tech_data = normallize_all(tech_data[-context.max_sequence_length:])
        index_features[i] = tech_data
    index_features = pd.Panel(index_features)
    assert stock_features.shape[0] == len(context.assets)
    assert stock_features.shape[1] == index_features.shape[1]
    
    # if context.i == 1:
    #     real_return = np.ones(stock_features.shape[0])
    #     context.real_return = list(real_return)
    # else:
    #     real_return = np.array(context.real_return)
    # real_return = pd.Series(real_return, name='real_return', index=stock_features.iloc[0].index)
    # real_return = np.log((real_return / real_return.shift(1)).fillna(1))
    
    return_rate = (prices / prices.shift(1)).join(pd.Series(np.ones(prices.shape[0]) * 1.001, index=prices.index, name='CASH'))[stock_features.iloc[0].index[0]:]
    
    spy_return = (spy_index / spy_index.shift(1))[stock_features.iloc[0].index[0]:]
    feed = context.model.build_feed_dict(input_data={'equity_network': stock_features.values, 'index_network': index_features.values}, return_rate=return_rate, fee=context.model_fee, keep_prob=1.0)
    if context.i == 1:
        context.model.train(feed)
    rewards, cum_log_reward, cum_reward, actions = context.model.trade(feed)
    print('actual return', context.portfolio.returns + 1, 'expect reward:', cum_reward, 'on', str(trading_date))
    record(predict_reward=cum_reward.ravel()[0])
    record(spy=spy_index[-1])
    record(spy_return=spy_return.prod() - 1)
    context.today_action = actions[-1].flatten()
    record(invest_weight=np.sum(context.today_action))
    if context.i % context.model_update_time == 0:
        context.model.train(feed=feed)
        rewards, cum_log_reward, cum_reward, actions = context.model.trade(feed)
        epoch = 0
        while cum_reward < 1.2 and epoch < 10:
            feed = context.model.change_drop_keep_prob(feed, 0.8)
            context.model.train(feed=feed)
            rewards, cum_log_reward, cum_reward, actions = context.model.trade(feed)
            epoch += 1


def handle_data(context, data):
    context.i += 1
    holding_securities = dict(filter(lambda x: x[1] > 0.2, list(zip(context.assets, context.today_action))))
    print(holding_securities)
    action = context.today_action
    for k, asset in enumerate(context.assets):
        order_target_percent(asset, action[k])
    if context.tensorboard is not None:
        context.tensorboard.log_algo(context, epoch=context.i)
    context.real_return.append(context.portfolio.returns)


if __name__ == '__main__':
    back_test_name = 'model_EIIE_attention2'
    if not os.path.exists('sp500.csv'):
        print('downloading sp500 data')
        with open('sp500.csv', 'wb+') as f:
            response = requests.get('https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
            f.write(response.content)
    sp500 = pd.read_csv('sp500.csv')
    sp500.index = sp500['Symbol']
    high_cap_company = sp500.loc[list(itertools.chain.from_iterable(list(map(lambda x: x[1][-5:], list(sp500.sort_values('Market Cap').groupby('Sector').groups.items())))))]
    
    start = pd.Timestamp(pd.to_datetime('2005-02-08')).tz_localize('US/Eastern')
    end = pd.Timestamp(pd.to_datetime('2018-03-27')).tz_localize('US/Eastern')
    result = zipline.run_algorithm(start=start, end=end,
                                   initialize=initialize,
                                   before_trading_start=before_trading_start,
                                   handle_data=handle_data,
                                   capital_base=100000,
                                   data_frequency='daily',
                                   bundle='quandl'
                                   )
    result.to_pickle(back_test_name)
