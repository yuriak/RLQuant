# -*- coding:utf-8 -*-
import itertools
import os
import sys

import logbook
import numpy as np
import pandas as pd
import quandl
import requests
import talib
import zipline
from zipline.api import record, symbol, order_target_percent
from zipline.data import bundles
from zipline.finance import commission, slippage

from history.DRL_Portfolio import DRL_Portfolio

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(level=logbook.DEBUG),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()
from history.ZiplineTensorboard import TensorBoard

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
    data = data.dropna()
    return data


z_score = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)


def batch_nomorlize(f_data):
    need_normalize = f_data.columns[list(f_data.columns.map(lambda x: '_' in x))]
    keep_original = f_data.columns[list(f_data.columns.map(lambda x: '_' not in x))]
    return z_score(f_data[need_normalize]).join(f_data[keep_original])


def initialize(context):
    context.i = 0
    context.assets = list(map(lambda x: symbol(x), high_cap_company.Symbol.values))
    print(context.assets, len(context.assets))
    context.model_fee = 1e-3
    context.previous_predict_reward = 0
    context.previous_action = 0
    context.set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    context.bootstrap_sequence_length = 300
    context.tb_log_dir = './log/%s' % back_test_name
    context.tensorboard = TensorBoard(log_dir=context.tb_log_dir)
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
        history_data = history_data.transpose(2, 1, 0)
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
        index_data = index_data.transpose(2, 1, 0)
        index_data.to_pickle('index')
        context.index_data = index_data['Last', str(initial_history_start_date):]
    else:
        print('index data exist')
        index_data = pd.read_pickle('index')
        context.index_data = index_data['Last', str(initial_history_start_date):]
    context.model = DRL_Portfolio(feature_number=len(context.assets) * 8 + context.index_data.columns.shape[0] * 7, asset_number=len(context.assets) + 1, object_function='sortino')
    context.model.init_model()


def before_trading_start(context, data):
    trading_date = context.get_datetime().date()
    prices = context.history_data['adj_close'][:str(trading_date)][:-1]
    volumes = context.history_data['adj_volume'][:str(trading_date)][:-1]
    index_data = context.index_data[:str(trading_date)][:-1]
    prices = prices.fillna(1)
    volumes = volumes.fillna(1)
    symbols = list(prices.columns)
    volumes = np.log(volumes)
    volumes = volumes.rename(columns=lambda x: x + '_log_volume')
    full_features = pd.concat(tuple([generate_tech_data(prices[c].astype(float)) for c in symbols]), axis=1)
    index_features = pd.concat(tuple([generate_tech_data(index_data[i].astype(float)) for i in index_data.columns]))
    index_return = index_data / index_data.shift(1)[full_features.index[0]:]
    log_index_return = np.log(index_return)
    return_rate = (prices / prices.shift(1))[full_features.index[0]:]
    log_return_rate = np.log(return_rate)
    f_data = full_features.join(prices).join(volumes).join(index_features).join(index_data)
    z_data = return_rate.join(pd.Series(np.ones((f_data.shape[0])) * 1.0001, index=f_data.index, name='ASSET'))[f_data.index[0]:]
    hidden_initial_state, current_rnn_output = context.model.get_rnn_zero_state()
    feed = context.model.build_feed_dict(batch_F=batch_nomorlize(f_data),
                                         batch_Z=z_data,
                                         keep_prob=1.0,
                                         fee=context.model_fee,
                                         rnn_hidden_init_state=hidden_initial_state,
                                         output_hidden_init_state=current_rnn_output,
                                         initial_output=current_rnn_output)
    rewards, cum_log_reward, cum_reward, actions, hidden_initial_state, output_initial_state, current_rnn_output = context.model.trade(feed)
    real_multiplier = context.target_profit_multiplier * (1 + context.i * 0.001)
    # target_return = index_return['spy'].prod() * real_multiplier
    index_current_return = index_return['spy'].prod()
    target_return = context.target_profit_multiplier * 1.5 * index_current_return if index_current_return < 1 else context.target_profit_multiplier * index_current_return
    
    while cum_reward < target_return:
        feed = context.model.change_drop_keep_prob(feed, 0.8)
        context.model.train(feed=feed)
        feed = context.model.change_drop_keep_prob(feed, 1.0)
        rewards, cum_log_reward, cum_reward, actions, hidden_initial_state, output_initial_state, current_rnn_output = context.model.trade(feed)
    context.today_action = actions[-1].flatten()
    record(predict_reward=cum_reward.ravel()[0])
    record(spy=index_data['spy'][-1])
    record(spy_return=index_return['spy'].prod() - 1)


def handle_data(context, data):
    # trading_date = data.history(context.assets, ['close'], bar_count=1, frequency='1d').index[0].date()
    trading_date = context.get_datetime().date()
    context.i += 1
    action = context.today_action.flatten()[:-1]
    for k, asset in enumerate(context.assets):
        order_target_percent(asset, action[k])
    if context.i % 100 == 0:
        print(trading_date)
        print(context.portfolio.portfolio_value)
    if context.tensorboard is not None:
        # record algo stats to tensorboard
        context.tensorboard.log_algo(context, epoch=context.i)


if __name__ == '__main__':
    back_test_name = 'price_index_sortino_backtest2'
    if not os.path.exists('sp500.csv'):
        print('downloading sp500 data')
        with open('sp500.csv', 'wb+') as f:
            response = requests.get('https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
            f.write(response.content)
    sp500 = pd.read_csv('sp500.csv')
    sp500.index = sp500['Symbol']
    high_cap_company = sp500.loc[list(itertools.chain.from_iterable(list(map(lambda x: x[1][-3:], list(sp500.sort_values('Market Cap').groupby('Sector').groups.items())))))]
    
    start = pd.Timestamp(pd.to_datetime('2005-02-08')).tz_localize('US/Eastern')
    end = pd.Timestamp(pd.to_datetime('2018-03-26')).tz_localize('US/Eastern')
    result = zipline.run_algorithm(start=start, end=end,
                                   initialize=initialize,
                                   before_trading_start=before_trading_start,
                                   handle_data=handle_data,
                                   capital_base=100000,
                                   data_frequency='daily',
                                   bundle='quandl'
                                   )
    result.to_pickle(back_test_name)
