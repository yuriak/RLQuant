# -*- coding:utf-8 -*-
import os
import requests
import itertools
from utils.EnvironmentUtils import build_backtest_environment
from utils.DataUtils import *
import tensorflow as tf
from model_archive.DRL_Portfolio_Isolated_Simple import DRL_Portfolio
from env.zipline_env import AgentTrader
import pickle

start_date_str = '2005-02-08'
end_date_str = '2018-03-27'
bootstrap_length = 300

data, env, bundle, sim_params = build_backtest_environment(start_date_str, end_date_str)
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

# ==============================================================================
# Start the backtest
# Step 1. define the hyper-parameter combination
training_sequence_length = [30, 60, 100, 150, None]
taos = [1.0, 5.0]
attention_length = [5, 10]
network_plan = [
    ([128], [64]),
    ([256, 128], [64, 32]),
    ([512, 256, 128], [128, 64])
]
object_function = ['reward', 'sortino']

equity_network_template = {
    'feature_map_number': len(assets),
    'feature_number': equity_data.shape[2],
    'input_name': 'equity',
    'keep_output': True
}

index_network_template = {
    'feature_map_number': len(index_data.items),
    'feature_number': index_data.shape[2],
    'input_name': 'index',
    'keep_output': False
}
news_network_template = {
    'feature_map_number': 1,
    'feature_number': 100,
    'input_name': 'news',
    'keep_output': False
}

weight_network_template = {
    'feature_map_number': 1,
    'feature_number': len(assets) + 1,
    'input_name': 'weight',
    'keep_output': False
}
return_network_template = {
    'feature_map_number': 1,
    'feature_number': 1,
    'input_name': 'return',
    'keep_output': False
}
networks = {
    'equity_network': equity_network_template,
    'weight_network': weight_network_template,
    'return_network': return_network_template,
    'index_network': index_network_template,
    'news_network': news_network_template
}
other_features = {
    'index_network': {
        'data': index_data,
        'normalize': True
    },
    'news_network': {
        'data': news_data,
        'normalize': False
    }
}
hyper_parameters = []
for d, r in network_plan:
    for act in [tf.nn.relu, tf.nn.tanh]:
        for attn in attention_length:
            for tao in taos:
                for sequence_length in training_sequence_length:
                    for o in object_function:
                        network_topology = {}
                        training_strategy = {
                            'training_data_length': sequence_length,
                            'tao': tao,
                            'short_term': {
                                'interval': 1,
                                'max_epoch': 1,
                                'keep_prob': 1.0
                            },
                            'long_term': {
                                'interval': 30,
                                'max_epoch': 10,
                                'keep_prob': 0.85,
                            }
                        }
                        for k, v in networks.items():
                            template = v
                            if k == 'equity_network':
                                template['dense'] = {
                                    'n_units': d,
                                    'act': [act] * len(d)
                                }
                                template['rnn'] = {
                                    'n_units': r + [1],
                                    'act': [act] * len(r) + [tf.nn.sigmoid],
                                    'attention_length': attn
                                }
                            else:
                                template['dense'] = {
                                    'n_units': d,
                                    'act': [act] * len(d)
                                }
                                template['rnn'] = {
                                    'n_units': r,
                                    'act': [act] * len(r),
                                    'attention_length': attn
                                }
                            network_topology[k] = template
                        hyper_parameters.append((network_topology, training_strategy, o))

if not os.path.exists('./experiment'):
    os.mkdir('./experiment')

for i, h in enumerate(hyper_parameters):
    result_dir = './experiment/result%d' % i
    model_dir = result_dir + '/model_archive'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    topology = h[0]
    strategy = h[1]
    o_function = h[2]
    model = DRL_Portfolio(asset_number=len(assets), feature_network_topology=topology, object_function=o_function, learning_rate=0.001)
    trader = AgentTrader(
        model=model,
        pre_defined_assets=assets,
        equity_data=equity_data,
        other_data=other_features,
        training_strategy=strategy,
        sim_params=sim_params,
        pre_trained_model_path=None,
        name='backtest_%d' % (i),
        env=env
    )
    try:
        with open(result_dir + '/hyper_parameter', 'wb+') as f:
            pickle.dump({'topology': topology, 'strategy': strategy, 'object': o_function}, file=f)
        trained_model, actions, result = trader.backtest(data)
        trained_model.save_model(model_dir)
        np.save(result_dir + '/action', actions)
        result.to_pickle(result_dir + '/result')
    except Exception as e:
        print(e.message)
        continue
