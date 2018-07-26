# -*- coding:utf-8 -*-
import re
import tensorflow as tf

import requests
import itertools

from zipline.data import bundles
from zipline.utils.calendars import get_calendar
from zipline.finance.trading import TradingEnvironment
from zipline.utils.factory import create_simulation_parameters
from zipline.data.data_portal import DataPortal

from env.zipline_env import AgentTrader
from model_archive.DRL_Portfolio_Isolated_Simple import DRL_Portfolio
from utils.DataUtils import *
from utils.EnvironmentUtils import *

start_date_str = '2005-02-08'
end_date_str = '2018-03-27'
capital_base=100000
bootstrap_length = 300
data, env, bundle, sim_params=build_backtest_environment(start_date_str,end_date_str,capital_base=capital_base)
# =========================================
# load security pool
if not os.path.exists('sp500.csv'):
    print('downloading sp500 data')
    with open('sp500.csv', 'wb+') as f:
        response = requests.get('https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
        f.write(response.content)
sp500 = pd.read_csv('sp500.csv')
sp500.index = sp500['Symbol']
high_cap_company = sp500.loc[list(itertools.chain.from_iterable(list(map(lambda x: x[1][:5], list(sp500.sort_values('Market Cap').groupby('Sector').groups.items())))))]
assets = list(high_cap_company.Symbol.values)
assets = retrieve_equitys(bundle, assets)
# =========================================
# prepare data
initial_history_start_date = bundle.equity_daily_bar_reader.sessions[bundle.equity_daily_bar_reader.sessions < start_date_str][(-bootstrap_length - 1)]
initial_history_end_date = bundle.equity_daily_bar_reader.sessions[bundle.equity_daily_bar_reader.sessions > start_date_str][0]
filtered_assets_index = (np.isnan(np.sum(bundle.equity_daily_bar_reader.load_raw_arrays(columns=['close'], start_date=initial_history_start_date, end_date=initial_history_end_date, assets=assets), axis=1)).flatten() == False)
assets = list(np.array(assets)[filtered_assets_index])
print(assets, len(assets))
remain_asset_names = list(map(lambda x: x.symbol, assets))

equity_data = prepare_equity_data(initial_history_start_date, remain_asset_names)
index_data = prepare_index_data(initial_history_start_date, equity_data.major_axis)
news_data = prepare_news_data(equity_data)

# The dictionary may change the order of assets, so we rebuild the assets list
assets = retrieve_equitys(bundle, list(equity_data.items))
remain_asset_names = list(map(lambda x: x.symbol, assets))

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
            'act': [tf.nn.tanh, tf.nn.tanh],
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
            'n_units': [2, 1],
            'act': [tf.nn.tanh, tf.nn.tanh],
            'attention_length': 10
        },
        'keep_output': False
    },
    # 'news_network': {
    #     'feature_map_number': 1,
    #     'feature_number': 100,
    #     'input_name': 'return',
    #     'dense': {
    #         'n_units': [128, 64],
    #         'act': [tf.nn.tanh] * 2,
    #     },
    #     'rnn': {
    #         'n_units': [32, 16],
    #         'act': [tf.nn.tanh, tf.nn.tanh],
    #         'attention_length': 10
    #     },
    #     'keep_output': False
    # }
}

other_features = {
    # 'news_network': {
    #     'data': news_data,
    #     'normalize': False
    # },
    'index_network': {
        'data': index_data,
        'normalize': True
    }
}

training_strategy = {
    'training_data_length': 30,
    'tao': 5.0,
    'short_term': {
        'interval': 1,
        'max_epoch': 1,
        'keep_prob': 1.0,
    },
    'long_term': {
        'interval': 30,
        'max_epoch': 10,
        'keep_prob': 0.85,
        'target_reward': 1.2
    }
}

model = DRL_Portfolio(asset_number=len(assets), feature_network_topology=network_topology, object_function='reward', learning_rate=0.001)
trader = AgentTrader(model=model,
                     pre_defined_assets=assets,
                     equity_data=equity_data,
                     other_data=other_features,
                     training_strategy=training_strategy,
                     sim_params=sim_params, env=env)
trained_model, actions, result = trader.backtest(data)
trained_model.save_model()
np.save('actions', actions)
result.to_pickle('trading_result')
