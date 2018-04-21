# -*- coding:utf-8 -*-
import os

from zipline.data import bundles
from zipline.utils.calendars import get_calendar
from zipline.finance.trading import TradingEnvironment
from zipline.utils.factory import create_simulation_parameters
from zipline.data.data_portal import DataPortal
import pandas as pd
import re

def build_backtest_environment(star_date, end_date,capital_base=100000):
    trading_calendar = get_calendar("NYSE")
    sim_params = create_simulation_parameters(capital_base=capital_base,
                                              data_frequency='daily',
                                              trading_calendar=trading_calendar,
                                              start=pd.Timestamp(pd.to_datetime(star_date)).tz_localize('US/Eastern'),
                                              end=pd.Timestamp(pd.to_datetime(end_date)).tz_localize('US/Eastern')
                                              )
    bundle = bundles.load('quandl')
    prefix, connstr = re.split(r'sqlite:///', str(bundle.asset_finder.engine.url), maxsplit=1, )
    env = TradingEnvironment(asset_db_path=connstr, environ=os.environ)
    data = DataPortal(
        env.asset_finder, trading_calendar,
        first_trading_day=bundle.equity_minute_bar_reader.first_trading_day,
        equity_minute_reader=bundle.equity_minute_bar_reader,
        equity_daily_reader=bundle.equity_daily_bar_reader,
        adjustment_reader=bundle.adjustment_reader,
    )
    return data,env,bundle,sim_params