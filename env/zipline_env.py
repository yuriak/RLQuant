# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import zipline
from zipline.api import record, symbol, order_target_percent
from zipline.data import bundles
from zipline.finance import commission, slippage
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.calendars import get_calendar
from zipline import TradingAlgorithm
from zipline.data.data_portal import DataPortal
from zipline.finance.trading import TradingEnvironment
from utils.DataUtils import normalize_all
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
from zipline.data import bundles
from utils.ZiplineTensorboard import TensorBoard


# other_features = {
#     'news_network': {
#         'data': None,
#         'normalize': False
#     },
#     'index_network': {
#         'data': None,
#         'normalize': True
#     }
# }
# trading_stategy = {
#     'training_data_length': 100,
#     'tao': 10.0,
#     'short_term': {
#         'interval': 1,
#         'max_epoch': 1,
#         'keep_prob': 0.95,
#     },
#     'long_term': {
#         'interval': 10,
#         'max_epoch': 10,
#         'keep_prob': 0.8,
#         'target_reward': 1.2
#     }
# }


class AgentTrader(TradingAlgorithm):
    def __init__(self, model, pre_defined_assets, equity_data, other_data, training_strategy, pre_trained_model_path=None, name='backtest', log_interval=1, transaction_cost=0.005, *args, **kwargs):
        TradingAlgorithm.__init__(self, *args, **kwargs)
        self.model = model
        self.assets = pre_defined_assets
        self.transaction_cost = transaction_cost
        self.training_strategy = training_strategy
        self.other_training_data = other_data
        self.equity_data = equity_data
        self.log_dir = 'log/' + name
        self.log_interval = log_interval
        self.real_return = []
        self.history_weight = []
        if pre_trained_model_path == None:
            self.model.init_model()
        else:
            self.model.load_model(pre_trained_model_path)
        self.day = 1
        self.backtest_action_record = []
        self.tensorboard = TensorBoard(log_dir=self.log_dir, session=self.model.get_session())
    
    def initialize(self):
        self.set_commission(commission.PerShare(cost=self.transaction_cost, min_trade_cost=1.0))
        self.set_slippage(slippage.VolumeShareSlippage())
    
    def handle_data(self, data):
        trading_date = self.get_datetime().date()
        training_data_length = self.training_strategy['training_data_length']
        # =======================================================================================
        # Prepare data
        if training_data_length is not None:
            equity_features = self.equity_data[:, :str(trading_date), :][:, -training_data_length - 1:-1, :]
        else:
            equity_features = self.equity_data[:, :str(trading_date), :][:, :-1, :]
        print(equity_features.major_axis[0], equity_features.major_axis[-1])
        if self.day == 1:
            real_return = np.zeros(equity_features.shape[1])
            self.real_return = list(real_return)
        else:
            real_return = np.array(self.real_return)[-equity_features.shape[1]:]
        
        if self.day == 1:
            portfolio_weight = np.ones((equity_features.shape[1], len(self.assets)))
            portfolio_weight = np.exp(portfolio_weight) / (np.sum(np.exp(portfolio_weight), axis=1).reshape((equity_features.shape[1], 1)))
            self.history_weight = list(portfolio_weight)
        else:
            portfolio_weight = np.array(self.history_weight)[-equity_features.shape[1]:]
        return_features = np.expand_dims(real_return.reshape(real_return.shape[0], 1), axis=0)
        portfolio_weight_features = np.expand_dims(portfolio_weight, axis=0)
        
        assert return_features.shape[1] == equity_features.shape[1]
        assert portfolio_weight_features.shape[1] == equity_features.shape[1]
        assert portfolio_weight_features.shape[2] == len(self.assets)
        
        return_rate = equity_features[:, :, 'return_rate'].join(pd.Series(np.ones(equity_features.shape[1]) * 1.001, index=equity_features.major_axis, name='CASH'))
        equity_features = equity_features.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0)
        assert np.sum(np.isnan(equity_features.values)) == 0
        assert np.sum(np.isnan(portfolio_weight_features)) == 0
        assert np.sum(np.isnan(return_features)) == 0
        assert np.sum(np.isnan(return_rate.values)) == 0
        assert np.sum(return_rate.values <= 0) == 0
        input_data = {'equity_network': equity_features.values, 'weight_network': portfolio_weight_features, 'return_network': return_features}
        for k, v in self.other_training_data.items():
            other_feature = v['data']
            if len(other_feature.shape) > 2:
                other_feature = other_feature[:, equity_features.major_axis, :].fillna(0)
                other_feature = other_feature.replace(-np.inf,np.nan).replace(np.inf, np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0)
                other_feature = other_feature.values
            else:
                other_feature = other_feature.loc[equity_features.major_axis].fillna(0)
                other_feature.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0)
                other_feature = np.expand_dims(other_feature.values, axis=0)
            assert equity_features.shape[1] == other_feature.shape[1]
            assert np.sum(np.isnan(other_feature)) == 0
            input_data[k] = other_feature
        
        assert return_rate.shape[0] == equity_features.shape[1]
        feed = self.model.build_feed_dict(input_data=input_data,
                                          return_rate=return_rate.values,
                                          fee=self.transaction_cost,
                                          keep_prob=1.0,
                                          tao=self.training_strategy['tao'])
        
        # self.model_archive.train(feed)
        # if self.day >30:
        # =====================================================================================
        # Conduct shor term training, for example, daily update model_archive
        if 'short_term' in self.training_strategy.keys():
            training_strategy = self.training_strategy['short_term']
            if self.day % training_strategy['interval'] == 0:
                feed = self.model.change_drop_keep_prob(feed, training_strategy['keep_prob'])
                for _ in range(training_strategy['max_epoch']):
                    self.model.train(feed)
        
        # ==================================================================================================
        # Execute Orders
        rewards, cum_log_reward, cum_reward, actions = self.model.trade(feed)
        today_action = np.nan_to_num(actions[-1].flatten())
        if self.day % self.training_strategy['execute_interval'] == 0:
            for k, asset in enumerate(self.assets):
                order_target_percent(asset, today_action[k])
        current_portfolio_value = self.portfolio.portfolio_value
        real_portfolio_weight = list(map(lambda x: self.portfolio.positions[x].amount * self.portfolio.positions[x].last_sale_price / current_portfolio_value, self.assets))
        real_portfolio_weight = np.array(real_portfolio_weight)
        self.real_return.append((self.portfolio.returns - self.real_return[-1]))
        self.history_weight.append(real_portfolio_weight)
        self.backtest_action_record.append(today_action)
        holding_securities = dict(filter(lambda x: x[1] > 0.02 or x[1] < -0.02, list(zip(self.assets, today_action))))
        if self.day % self.log_interval == 0:
            record(invest_weight=np.sum(np.abs(today_action[:-1])))
            record(predict_reward=cum_reward.ravel()[0])
            record(large_holding=len(holding_securities))
            record(long_order_number=np.sum(real_portfolio_weight > 0))
            record(short_order_number=np.sum(real_portfolio_weight < 0))
            model_summary = self.model.get_summary(feed)
            self.tensorboard.log_algo(self, model_summaries=model_summary, epoch=self.day)
        print('actual return', self.portfolio.returns + 1, 'expect return:', cum_reward, 'on', str(trading_date))
        print(holding_securities)
        print('=' * 100)
        
        # =======================================================================================
        # Conduct long term training, for example, monthly update model_archive
        if 'long_term' in self.training_strategy.keys():
            training_strategy = self.training_strategy['long_term']
            if self.day % training_strategy['interval'] == 0:
                feed = self.model.change_drop_keep_prob(feed, training_strategy['keep_prob'])
                epoch = 0
                while epoch < training_strategy['max_epoch']:
                    self.model.train(feed)
                    epoch += 1
        self.day += 1
    
    
    def backtest(self, data):
        result = self.run(data)
        return self.model, np.array(self.backtest_action_record), result
