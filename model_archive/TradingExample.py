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
from model_archive.DRL_Portfolio_Highway import DRL_Portfolio
from utils.DataUtils import *
from utils.EnvironmentUtils import *

start_date_str = '2005-02-08'
end_date_str = '2018-03-27'
capital_base = 100000
bootstrap_length = 300
data, env, bundle, sim_params = build_backtest_environment(start_date_str, end_date_str, capital_base=capital_base)
# =========================================
# load security pool
if not os.path.exists('sp500.csv'):
    print('downloading sp500 data')
    with open('sp500.csv', 'wb+') as f:
        response = requests.get('https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
        f.write(response.content)
sp500 = pd.read_csv('sp500.csv')
sp500.index = sp500['Symbol']
high_cap_company = sp500.loc[list(itertools.chain.from_iterable(list(map(lambda x: x[1][-20:-10], list(sp500.sort_values('Market Cap').groupby('Sector').groups.items())))))]
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
spy_data = pd.Panel({'spy': index_data['spy', :, :]})
vix_data = pd.Panel({'vix': index_data['vix', :, :]})
gc_data = pd.Panel({'gc': index_data['gc', :, :]})
si_data = pd.Panel({'si': index_data['si', :, :]})
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
            'n_units': [equity_data.shape[2] * len(assets) // 3] * 5,
            'act': [tf.nn.relu] * 4,
        },
        'rnn': {
            'n_units': [256, 128, len(assets)],
            'act': [tf.nn.relu, tf.nn.relu, tf.nn.tanh],
            'attention_length': 10
        },
        'normalize': True,
        'keep_output': True
    },
    'spy_network': {
        'feature_map_number': 1,
        'feature_number': index_data.shape[2],
        'input_name': 'index',
        'dense': {
            'n_units': [index_data.shape[2]] * 5,
            'act': [tf.nn.relu] * 5,
        },
        'rnn': {
            'n_units': [index_data.shape[2], 64, 32],
            'act': [tf.nn.relu] * 3,
            'attention_length': 10
        },
        'normalize': True,
        'keep_output': False
    },
    'vix_network': {
        'feature_map_number': 1,
        'feature_number': index_data.shape[2],
        'input_name': 'index',
        'dense': {
            'n_units': [index_data.shape[2]] * 5,
            'act': [tf.nn.relu] * 5,
        },
        'rnn': {
            'n_units': [index_data.shape[2], 64, 32],
            'act': [tf.nn.relu] * 3,
            'attention_length': 10
        },
        'normalize': True,
        'keep_output': False
    },
    'gc_network': {
        'feature_map_number': 1,
        'feature_number': index_data.shape[2],
        'input_name': 'index',
        'dense': {
            'n_units': [index_data.shape[2]] * 5,
            'act': [tf.nn.relu] * 5,
        },
        'rnn': {
            'n_units': [index_data.shape[2], 64, 64],
            'act': [tf.nn.relu] * 3,
            'attention_length': 10
        },
        'normalize': True,
        'keep_output': False
    },
    'si_network': {
        'feature_map_number': 1,
        'feature_number': index_data.shape[2],
        'input_name': 'index',
        'dense': {
            'n_units': [index_data.shape[2]] * 5,
            'act': [tf.nn.relu] * 5,
        },
        'rnn': {
            'n_units': [index_data.shape[2], 64, 64],
            'act': [tf.nn.relu] * 3,
            'attention_length': 10
        },
        'normalize': True,
        'keep_output': False
    },
    'weight_network': {
        'feature_map_number': 1,
        'feature_number': len(assets),
        'input_name': 'weight',
        'dense': {
            'n_units': [(len(assets))] * 5,
            'act': [tf.nn.relu] * 5,
        },
        'rnn': {
            'n_units': [(len(assets)), 64],
            'act': [tf.nn.relu, tf.nn.relu],
            'attention_length': 10
        },
        'normalize': False,
        'keep_output': False
    },
    'return_network': {
        'feature_map_number': 1,
        'feature_number': 1,
        'input_name': 'return',
        'dense': {
            'n_units': [128] * 4,
            'act': [tf.nn.relu] * 4,
        },
        'rnn': {
            'n_units': [128, 64],
            'act': [tf.nn.relu, tf.nn.relu],
            'attention_length': 10
        },
        'normalize': False,
        'keep_output': False
    },
    'news_network': {
        'feature_map_number': 1,
        'feature_number': 100,
        'input_name': 'return',
        'dense': {
            'n_units': [128] * 4,
            'act': [tf.nn.relu] * 4,
        },
        'rnn': {
            'n_units': [128, 64],
            'act': [tf.nn.relu, tf.nn.relu],
            'attention_length': 10
        },
        'normalize': False,
        'keep_output': False
    }
}

other_features = {
    'news_network': {
        'data': news_data,
    },
    'spy_network': {
        'data': spy_data,
    },
    'vix_network': {
        'data': vix_data,
    },
    'gc_network': {
        'data': gc_data,
    },
    'si_network': {
        'data': si_data,
    }
}

training_strategy = {
    'training_data_length': 7,
    'tao': 10.0,
    'short_term': {
        'interval': 1,
        'max_epoch': 1,
        'keep_prob': 0.95,
    },
    'long_term': {
        'interval': 15,
        'max_epoch': 5,
        'keep_prob': 0.90,
        'target_reward': 1.2
    },
    'execute_interval': 7
}

model = DRL_Portfolio(asset_number=len(assets), feature_network_topology=network_topology, object_function='sharpe', learning_rate=0.001)
trader = AgentTrader(model=model,
                     pre_defined_assets=assets,
                     equity_data=equity_data,
                     other_data=other_features,
                     training_strategy=training_strategy,
                     sim_params=sim_params, env=env, name='backtest_3')
trained_model, actions, result = trader.backtest(data)
trained_model.save_model('backtest3')
np.save('actions_3', actions)
result.to_pickle('trading_result_3')
# 4AUY1FEpfGtYutRShAsmTMbVFmLoZdL92Gg6fQPYsN1P61mqrZpgnmsQKtYM8CkFpvDMJS6MuuKmncHhSpUtRyEqGcNUht2