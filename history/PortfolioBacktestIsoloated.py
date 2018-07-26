# -*- coding:utf-8 -*-
import pandas as pd
import zipline
import quandl
from DRL_Portfolio_EIIE_simple import DRL_Portfolio
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


def generate_tech_data(stock, open_name, close_name, high_name, low_name):
    open_price = stock[open_name].values
    close_price = stock[close_name].values
    low_price = stock[low_name].values
    high_price = stock[high_name].values
    data = pd.DataFrame(stock)
    data['MOM'] = talib.MOM(close_price)
    data['SMA'] = talib.SMA(close_price)
    data['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
    data['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
    data['sine'], data['leadsine'] = talib.HT_SINE(close_price)
    data['inphase'], data['quadrature'] = talib.HT_PHASOR(close_price)
    data['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_price)
    data['SAREXT'] = talib.SAREXT(high_price, low_price)
    data['ADX'] = talib.ADX(high_price, low_price, close_price)
    data['ADXR'] = talib.ADX(high_price, low_price, close_price)
    data['APO'] = talib.APO(close_price)
    data['AROON_UP'], data['AROON_DOWN'] = talib.AROON(high_price, low_price)
    data['AROONOSC'] = talib.AROONOSC(high_price, low_price)
    data['BOP'] = talib.BOP(open_price, high_price, low_price, close_price)
    data['CCI'] = talib.CCI(high_price, low_price, close_price)
    data['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price)
    data['PLUS_DM'] = talib.PLUS_DM(high_price, low_price)
    data['PPO'] = talib.PPO(close_price)
    data['macd'], data['macd_sig'], data['macd_hist'] = talib.MACD(close_price)
    data['RSI'] = talib.RSI(close_price)
    data['CMO'] = talib.CMO(close_price)
    data['ROC'] = talib.ROC(close_price)
    data['ROCP'] = talib.ROCP(close_price)
    data['ROCR'] = talib.ROCR(close_price)
    data['slowk'], data['slowd'] = talib.STOCH(high_price, low_price, close_price)
    data['fastk'], data['fastd'] = talib.STOCHF(high_price, low_price, close_price)
    data['TRIX'] = talib.TRIX(close_price)
    data['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price)
    data['WILLR'] = talib.WILLR(high_price, low_price, close_price)
    data['NATR'] = talib.NATR(high_price, low_price, close_price)
    data['TRANGE'] = talib.TRANGE(high_price, low_price, close_price)
    data = data.drop([open_name, close_name, high_name, low_name], axis=1)
    data = data.dropna()
    return data


z_score = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)


def batch_nomorlize(f_data):
    need_normalize = f_data.columns[list(f_data.columns.map(lambda x: '_' in x))]
    keep_original = f_data.columns[list(f_data.columns.map(lambda x: '_' not in x))]
    return z_score(f_data[need_normalize]).join(f_data[keep_original])


def normalize_all(f_data):
    return z_score(f_data)


def generate_stock_features(history_data):
    stock_features = {}
    for c in history_data.items:
        columns = ['adj_open', 'adj_close', 'adj_high', 'adj_low', 'adj_volume']
        stock_data = history_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
        tech_data = generate_tech_data(stock_data.astype(float), columns[0], columns[1], columns[2], columns[3])
        stock_data['log_volume'] = np.log(stock_data['adj_volume'])
        tech_data = tech_data.join(stock_data['log_volume'])
        return_rate = pd.Series((stock_data['adj_close'] / stock_data['adj_close'].shift(1)).fillna(1), name='return_rate')
        tech_data = tech_data.join(return_rate)
        stock_features[c] = tech_data
    return pd.Panel(stock_features)


def generate_index_features(index_data):
    index_features = {}
    for c in index_data.items:
        columns = ['Open', 'Last', 'High', 'Low']
        index = index_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
        tech_data = generate_tech_data(index.astype(float), columns[0], columns[1], columns[2], columns[3])
        return_rate = pd.Series((index['Last'] / index['Last'].shift(1)).fillna(1), name='return_rate')
        tech_data = tech_data.join(return_rate)
        index_features[c] = tech_data
    return pd.Panel(index_features)


def initialize(context):
    context.set_benchmark(None)
    context.i = 1
    context.assets = list(map(lambda x: symbol(x), high_cap_company.Symbol.values))
    print(context.assets, len(context.assets))
    context.model_fee = 5e-3
    context.set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    context.bootstrap_sequence_length = 300
    context.max_sequence_length = 60
    context.tb_log_dir = './log/%s' % back_test_name
    context.model_update_time = 30
    context.target_profit_multiplier = 1.1
    context.model_summaries = None
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
        history_data.to_pickle('history_data')
        context.history_data = generate_stock_features(history_data)
        print('Done')
    else:
        print('history data exist')
        history_data = pd.read_pickle('history_data')
        context.history_data = generate_stock_features(history_data)
    
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
        index_data.to_pickle('index')
        index_data = index_data[:, str(initial_history_start_date):, :]
        context.index_data = generate_index_features(index_data)
    else:
        print('index data exist')
        index_data = pd.read_pickle('index')
        index_data = index_data[:, str(initial_history_start_date):, :]
        context.index_data = generate_index_features(index_data)[:, context.history_data.major_axis[0]:, :]
    
    if not os.path.exists('trading_content'):
        sys.exit(1)
    else:
        news_vec = pd.read_csv('trading_content')
        news_vec.index = news_vec.date
        news_vec = news_vec.drop('date', axis=1)
        news_vec = context.history_data[:, :, 'return_rate'].join(news_vec).drop(context.history_data.items, axis=1).fillna(0)
        context.news_vec = news_vec
    assert context.history_data.major_axis[0] == context.index_data.major_axis[0]
    
    feature_network_topology = {
        'equity_network': {
            'feature_map_number': len(context.assets),
            'feature_number': context.history_data.shape[2],
            'input_name': 'equity',
            'dense': {
                'n_units': [128, 64],
                'act': [tf.nn.tanh] * 2,
            },
            'rnn': {
                'n_units': [32, 1],
                'act': [tf.nn.tanh, None],
                'attention_length': 10
            },
            'keep_output': True
        },
        'index_network': {
            'feature_map_number': len(context.index_data.items),
            'feature_number': context.index_data.shape[2],
            'input_name': 'index',
            'dense': {
                'n_units': [128, 64],
                'act': [tf.nn.tanh] * 2,
            },
            'rnn': {
                'n_units': [32, 16],
                'act': [tf.nn.tanh, tf.nn.tanh],
                'attention_length': 10
            },
            'keep_output': False
        },
        'weight_network': {
            'feature_map_number': 1,
            'feature_number': len(context.assets) + 1,
            'input_name': 'weight',
            'dense': {
                'n_units': [32, 16],
                'act': [tf.nn.tanh] * 2,
            },
            'rnn': {
                'n_units': [16, 8],
                'act': [tf.nn.tanh, tf.nn.tanh],
                'attention_length': 10
            },
            'keep_output': False
        },
        'return_network': {
            'feature_map_number': 1,
            'feature_number': 1,
            'input_name': 'return',
            'dense': {
                'n_units': [8, 4],
                'act': [tf.nn.tanh] * 2,
            },
            'rnn': {
                'n_units': [4, 2],
                'act': [tf.nn.tanh, tf.nn.tanh],
                'attention_length': 10
            },
            'keep_output': False
        },
        'news_network': {
            'feature_map_number': 1,
            'feature_number': 100,
            'input_name': 'return',
            'dense': {
                'n_units': [128, 64],
                'act': [tf.nn.tanh] * 2,
            },
            'rnn': {
                'n_units': [32, 16],
                'act': [tf.nn.tanh, tf.nn.tanh],
                'attention_length': 10
            },
            'keep_output': False
        }
    }
    context.model = DRL_Portfolio(asset_number=len(context.assets),
                                  feature_network_topology=feature_network_topology,
                                  action_network_layers=[32, 16],
                                  object_function='reward')
    context.real_return = []
    context.history_weight = []
    context.model.init_model()
    context.tensorboard = TensorBoard(log_dir=context.tb_log_dir, session=context.model.get_session())


def before_trading_start(context, data):
    trading_date = context.get_datetime().date()
    # prices = context.history_data[:, :str(trading_date), 'adj_close'][:-1]
    stock_features = context.history_data[:, :str(trading_date), :][:, -context.max_sequence_length - 1:-1, :]
    # stock_features = context.history_data[:, :str(trading_date), :]
    index_features = context.index_data[:, stock_features.major_axis, :]
    news_features = context.news_vec.loc[stock_features.major_axis].fillna(0)
    # spy_index = context.index_data['spy', :str(trading_date), 'Last'][:-1].fillna(method='ffill')
    assert stock_features.shape[1] == index_features.shape[1]
    
    if context.i == 1:
        real_return = np.zeros(stock_features.shape[1])
        context.real_return = list(real_return)
    else:
        real_return = np.array(context.real_return)[-stock_features.shape[1]:]
        # real_return = np.array(context.real_return)
    if context.i == 1:
        portfolio_weight = np.ones((stock_features.shape[1], len(context.assets) + 1))
        portfolio_weight = np.exp(portfolio_weight) / np.sum(np.exp(portfolio_weight), axis=1).reshape((stock_features.shape[1], 1))
        context.history_weight = list(portfolio_weight)
    else:
        # portfolio_weight = np.array(context.history_weight)[-context.max_sequence_length:]
        portfolio_weight = np.array(context.history_weight)[-stock_features.shape[1]:]
    return_features = np.expand_dims(real_return.reshape(real_return.shape[0], 1), axis=0)
    portfolio_weight_features = np.expand_dims(portfolio_weight, axis=0)
    assert return_features.shape[1] == stock_features.shape[1]
    assert portfolio_weight_features.shape[1] == stock_features.shape[1]
    assert portfolio_weight_features.shape[2] == len(context.assets) + 1

    # news_features = context.news_vec[:str(trading_date)][-context.max_sequence_length:]
    news_features = np.expand_dims(news_features.values, axis=0)
    assert news_features.shape[1] == stock_features.shape[1]

    # return_rate = (prices / prices.shift(1)).join(pd.Series(np.ones(prices.shape[0]) * 1.001, index=prices.index, name='CASH'))[stock_features.major_index[0]:]
    return_rate = stock_features[:, :, 'return_rate'].join(pd.Series(np.ones(stock_features.shape[1]) * 1.001, index=stock_features.major_axis, name='CASH'))
    stock_features = stock_features.apply(func=normalize_all, axis='major_axis')
    index_features = index_features.apply(func=normalize_all, axis='major_axis')
    # spy_return = (spy_index / spy_index.shift(1))[stock_features.major_axis[0]:]
    feed = context.model.build_feed_dict(input_data={'equity_network': stock_features.values,
                                                     'index_network': index_features.values,
                                                     'return_network': return_features,
                                                     'weight_network': portfolio_weight_features,
                                                     'news_network': news_features
                                                     },
                                         return_rate=return_rate,
                                         fee=context.model_fee,
                                         keep_prob=0.8,
                                         tao=10.0
                                         )
    # if context.i == 1:
    # for i in range(10):
    context.model.train(feed)
    rewards, cum_log_reward, cum_reward, actions = context.model.trade(feed)
    try:
        context.model_summaries = context.model.get_summary(feed)
    except Exception as e:
        pass
    print('actual return', context.portfolio.returns + 1, 'expect reward:', cum_reward, 'on', str(trading_date))
    record(predict_reward=cum_reward.ravel()[0])
    # record(spy=spy_index[-1])
    # record(spy_return=spy_return.prod() - 1)
    context.today_action = actions[-1].flatten()
    record(invest_weight=np.sum(context.today_action))
    if context.i % context.model_update_time == 0:
        context.model.train(feed=feed)
        rewards, cum_log_reward, cum_reward, actions = context.model.trade(feed)
        epoch = 0
        while cum_reward < 1.2 and epoch < 10:
            # feed = context.model_archive.change_drop_keep_prob(feed, 0.9)
            context.model.train(feed=feed)
            rewards, cum_log_reward, cum_reward, actions = context.model.trade(feed)
            epoch += 1


def handle_data(context, data):
    context.i += 1
    holding_securities = dict(filter(lambda x: x[1] > 0.05, list(zip(context.assets, context.today_action))))
    print(holding_securities)
    action = context.today_action
    action = np.nan_to_num(action)
    for k, asset in enumerate(context.assets):
        order_target_percent(asset, action[k])
    if context.tensorboard is not None:
        context.tensorboard.log_algo(context, model_summaries=context.model_summaries, epoch=context.i)
    context.real_return.append(context.portfolio.returns)
    context.history_weight.append(action)


if __name__ == '__main__':
    back_test_name = 'model_EIIE_simple'
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
