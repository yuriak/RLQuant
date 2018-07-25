# -*- coding:utf-8 -*-
import talib
import pandas as pd
import os
import numpy as np

import quandl

quandl.ApiConfig.api_key = 'CTq2aKvtCkPPgR4L_NFs'

z_score = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)


def generate_tech_data_default(stock, open_name, close_name, high_name, low_name, volume_name='vol'):
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
    data = data.drop([open_name, close_name, high_name, low_name, 'amount', 'count'], axis=1)
    data = data.dropna().astype(np.float32)
    return data


def generate_tech_data(stock, open_name, close_name, high_name, low_name, max_time_window=10):
    open_price = stock[open_name].values
    close_price = stock[close_name].values
    low_price = stock[low_name].values
    high_price = stock[high_name].values
    data = pd.DataFrame(stock)
    data['MOM'] = talib.MOM(close_price, timeperiod=max_time_window)
    # data['_SMA'] = talib.SMA(close_price)
    data['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
    data['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
    data['sine'], data['leadsine'] = talib.HT_SINE(close_price)
    data['inphase'], data['quadrature'] = talib.HT_PHASOR(close_price)
    # data['_HT_TRENDMODE'] = talib.HT_TRENDMODE(close_price)
    # data['_SAREXT'] = talib.SAREXT(high_price, low_price)
    # data['_ADX'] = talib.ADX(high_price, low_price, close_price)
    data['ADXR'] = talib.ADXR(high_price, low_price, close_price, timeperiod=max_time_window)
    data['APO'] = talib.APO(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window)
    data['AROON_UP'], _ = talib.AROON(high_price, low_price, timeperiod=max_time_window)
    # data['_BOP'] = talib.BOP(open_price, high_price, low_price, close_price)
    data['CCI'] = talib.CCI(high_price, low_price, close_price, timeperiod=max_time_window)
    data['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=max_time_window)
    # data['_PLUS_DM'] = talib.PLUS_DM(high_price, low_price)
    data['PPO'] = talib.PPO(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window)
    data['macd'], data['macd_sig'], data['macd_hist'] = talib.MACD(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window, signalperiod=max_time_window // 2)
    data['CMO'] = talib.CMO(close_price, timeperiod=max_time_window)
    #     data['ROC'] = talib.ROC(close_price)
    data['ROCP'] = talib.ROCP(close_price, timeperiod=max_time_window)
    #     data['ROCR'] = talib.ROCR(close_price)
    #     data['slowk'], data['slowd'] = talib.STOCH(high_price, low_price, close_price)
    data['fastk'], data['fastd'] = talib.STOCHF(high_price, low_price, close_price)
    data['TRIX'] = talib.TRIX(close_price, timeperiod=max_time_window)
    data['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price, timeperiod1=max_time_window // 2, timeperiod2=max_time_window, timeperiod3=max_time_window * 2)
    data['WILLR'] = talib.WILLR(high_price, low_price, close_price, timeperiod=max_time_window)
    data['NATR'] = talib.NATR(high_price, low_price, close_price, timeperiod=max_time_window)
    # data['_TRANGE'] = talib.TRANGE(high_price, low_price, close_price)
    data = data.drop([open_name, close_name, high_name, low_name], axis=1)
    #     data.columns=data.columns.map(lambda x:x[1:])
    data = data.dropna().astype(np.float32)
    return data


def batch_nomorlize(f_data):
    need_normalize = f_data.columns[list(f_data.columns.map(lambda x: '_' in x))]
    keep_original = f_data.columns[list(f_data.columns.map(lambda x: '_' not in x))]
    return z_score(f_data[need_normalize]).join(f_data[keep_original])


normalize_all = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)


def generate_stock_features(history_data, max_time_window=10):
    stock_features = {}
    for c in history_data.items:
        columns = ['adj_open', 'adj_close', 'adj_high', 'adj_low', 'adj_volume']
        stock_data = history_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
        tech_data = generate_tech_data(stock_data.astype(float), columns[0], columns[1], columns[2], columns[3], max_time_window=max_time_window)
        stock_data['adj_volume'] = stock_data['adj_volume'].replace(0, np.nan).fillna(method='ffill')
        stock_data['log_volume'] = np.log(stock_data['adj_volume'])
        tech_data = tech_data.join(stock_data['log_volume'])
        tech_data = tech_data.drop('adj_volume', axis=1)
        return_rate = pd.Series((stock_data['adj_close'] / stock_data['adj_close'].shift(1)).fillna(1), name='return_rate')
        tech_data = tech_data.join(return_rate)
        stock_features[c] = tech_data
    return pd.Panel(stock_features).dropna()


def generate_index_features(index_data, max_time_window=10):
    index_features = {}
    for c in index_data.items:
        columns = ['Open', 'Last', 'High', 'Low']
        index = index_data[c, :, columns].fillna(method='ffill').fillna(method='bfill')
        tech_data = generate_tech_data(index.astype(float), columns[0], columns[1], columns[2], columns[3], max_time_window=max_time_window)
        return_rate = pd.Series((index['Last'] / index['Last'].shift(1)).fillna(1), name='return_rate')
        tech_data = tech_data.join(return_rate)
        index_features[c] = tech_data
    return pd.Panel(index_features).dropna()


def prepare_equity_data(start_date, instruments, data_path='./data/equity_data', max_time_window=10):
    if not os.path.exists(data_path):
        print('Start to download good history data')
        equity_data = {}
        for s in instruments:
            print('downloading', s)
            stock = quandl.get_table('WIKI/PRICES', date={'gte': str(start_date)}, ticker=s)
            stock.index = stock.date
            equity_data[s] = stock
        equity_data = pd.Panel(equity_data).fillna(method='ffill').fillna(method='bfill')
        equity_data.to_pickle(data_path)
        equity_data = generate_stock_features(equity_data, max_time_window=max_time_window)
        print('Done')
    else:
        print('equity data exist')
        equity_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
        equity_data = generate_stock_features(equity_data, max_time_window=max_time_window)
    assert np.sum(np.isnan(equity_data.values)) == 0
    return equity_data


def prepare_index_data(start_date, equity_reference_index=None, data_path='data/index_data', max_time_window=10):
    if not os.path.exists(data_path):
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
        index_data = pd.Panel({'vix': vix, 'gc': gc, 'si': si, 'spy': spy}).fillna(method='ffill').fillna(method='bfill')
        index_data.to_pickle(data_path)
        index_data = index_data[:, str(start_date):, :]
        index_data = generate_index_features(index_data, max_time_window=max_time_window)
        print('Done')
    else:
        print('index data exist')
        index_data = pd.read_pickle(data_path).fillna(method='ffill').fillna(method='bfill')
        index_data = index_data[:, str(start_date):, :]
        index_data = generate_index_features(index_data, max_time_window=max_time_window)
    if equity_reference_index is not None:
        index_data = index_data[:, equity_reference_index, :]
    assert np.sum(np.isnan(index_data.values)) == 0
    return index_data


def prepare_news_data(reference_equity_data, data_path='data/news.csv'):
    if not os.path.exists(data_path):
        return None
    else:
        news_vec = pd.read_csv(data_path)
        news_vec.index = news_vec.date
        news_vec = news_vec.drop('date', axis=1)
        news_vec = reference_equity_data[:, :, 'return_rate'].join(news_vec).drop(reference_equity_data.items, axis=1).fillna(0)
        return news_vec


def retrieve_equitys(bundle, assets):
    sids = bundle.asset_finder.sids
    all_assets = bundle.asset_finder.retrieve_all(sids)
    return list(filter(lambda x: x.symbol in assets, all_assets))
