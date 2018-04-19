# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import itertools
import os
import quandl
from zipline.data import bundles
from zipline.utils.calendars import get_calendar
from zipline.utils.factory import create_simulation_parameters
from zipline.data.data_portal import DataPortal
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    get_datetime,
    order,
    order_target_percent
)
from trading_environment.Trader import AgentTrader
from model.DRL_Portfolio_Isolated_Simple import DRL_Portfolio
from utils.DataUtils import *

start_date_str= '2005-02-08'
end_date_str= '2018-03-27'
bootstrap_length=300
trading_calendar = get_calendar("NYSE")
sim_params = create_simulation_parameters(capital_base=10000,
                                          data_frequency='daily',
                                          trading_calendar=trading_calendar,
                                          start=pd.Timestamp(pd.to_datetime(start_date_str)).tz_localize('US/Eastern'),
                                          end=pd.Timestamp(pd.to_datetime(end_date_str)).tz_localize('US/Eastern')
                                          )
bundle = bundles.load('quandl')
data = DataPortal(
    bundle.asset_finder, trading_calendar,
    first_trading_day=bundle.equity_daily_bar_reader.first_trading_day,
    equity_minute_reader=bundle.equity_minute_bar_reader,
    equity_daily_reader=bundle.equity_daily_bar_reader,
    adjustment_reader=bundle.adjustment_reader,
)

# =========================================
# load security pool
if not os.path.exists('sp500.csv'):
    print('downloading sp500 data')
    with open('sp500.csv', 'wb+') as f:
        response = requests.get('https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
        f.write(response.content)
sp500 = pd.read_csv('sp500.csv')
sp500.index = sp500['Symbol']
high_cap_company = sp500.loc[list(itertools.chain.from_iterable(list(map(lambda x: x[1][-5:], list(sp500.sort_values('Market Cap').groupby('Sector').groups.items())))))]
assets = list(high_cap_company.Symbol.values)
assets=retrieve_equitys(bundle,assets)
# =========================================
# prepare data
initial_history_start_date = bundle.equity_daily_bar_reader.sessions[bundle.equity_daily_bar_reader.sessions < start_date_str][(-bootstrap_length - 1)]
initial_history_end_date = bundle.equity_daily_bar_reader.sessions[bundle.equity_daily_bar_reader.sessions > start_date_str][0]
filtered_assets_index = (np.isnan(np.sum(bundle.equity_daily_bar_reader.load_raw_arrays(columns=['close'], start_date=initial_history_start_date, end_date=initial_history_end_date, assets=assets), axis=1)).flatten() == False)
assets = list(np.array(assets)[filtered_assets_index])
print(assets, len(assets))
remain_asset_names = list(map(lambda x: x.symbol, assets))

equity_data=prepare_equity_data(start_date_str,remain_asset_names)
index_data=prepare_index_data(start_date_str,equity_data.major_axis)
news_data=prepare_news_data(equity_data)

assert equity_data.major_axis[0] == index_data.major_axis[0]

network_topology = {
    'equity_network': {
        'feature_map_number': len(assets),
        'feature_number': equity_data.shape[2],
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
        'feature_map_number': len(index_data.items),
        'feature_number': index_data.shape[2],
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
        'feature_number': len(assets) + 1,
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

other_features = {
    'news_network': {
        'data': news_data,
        'normalize': False
    },
    'index_network': {
        'data': index_data,
        'normalize': True
    }
}

trading_stategy = {
    'training_data_length': 100,
    'tao': 10.0,
    'short_term': {
        'interval': 1,
        'max_epoch': 1,
        'keep_prob': 0.95,
    },
    'long_term': {
        'interval': 10,
        'max_epoch': 10,
        'keep_prob': 0.8,
        'target_reward': 1.2
    }
}



model=DRL_Portfolio(asset_number=len(assets),feature_network_topology=network_topology,object_function='sortino',learning_rate=0.001)
trader=AgentTrader(model=model,pre_defined_assets=assets,equity_data=equity_data,other_data=other_features,training_strategy=trading_stategy)
trader.backtest(data)