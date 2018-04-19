import datetime
import tensorflow as tf
import numpy as np

"""
This code was from: https://github.com/jimgoo/zipline-tensorboard
Thanks to Jimgoo
"""


class TensorBoard(object):
    """
    TensorBoard is a visualization tool provided with TensorFlow.
    This class can be used to record attributes from a running
    Zipline algorithm.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
        tensorboard --logdir=/full_path_to_your_logs
    Args:
        log_dir: the path of the directory where to save the log
            files to be parsed by tensorboard
        max_queue: Maximum number of summaries or events pending to be
                   written to disk before one of the 'add' calls block.
                   [default 10]
        flush_secs: How often, in seconds, to flush the added summaries
            and events to disk. [default 120]
    """
    
    def __init__(self, session, log_dir='./logs', max_queue=10, flush_secs=120):
        self.log_dir = log_dir
        # self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir,
                                            max_queue=max_queue,
                                            flush_secs=flush_secs,
                                            graph=session.graph
                                            )
    
    def log_dict(self, epoch, logs, model_summaries=None):
        """
        Writes a dictionary of simple named values to TensorBoard.
        Args:
            epoch: An integer representing time.
            logs: A dict containing what we want to log to TensorBoard.
        """
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, global_step=epoch)
        if model_summaries is not None:
            self.writer.add_summary(model_summaries, global_step=epoch)
        self.writer.flush()
    
    def log_algo(self, algo, model_summaries=None, epoch=None, other_logs={}):
        """
        Logs info about a Zipline algorithm as it's running.
        Args:
            epoch: An integer representing algorithm time.
                   If None, then the algorithm's current
                   date is converted to an ordinal so that
                   these integers are monotonically increasing
                   with time. The same integer convention should
                   be used across different runs so that their charts
                   line up correctly.
           algo: An instance of a zipline.algorithm.TradingAlgorithm
           other_logs: A dictionary containing other things we want to log.
        """
        if epoch is None:
            epoch = datetime.date.toordinal(algo.get_datetime())
        
        logs = {}
        
        # add portfolio related things
        logs['portfolio value'] = algo.portfolio.portfolio_value
        logs['portfolio pnl'] = algo.portfolio.pnl
        logs['portfolio return'] = algo.portfolio.returns
        logs['portfolio cash'] = algo.portfolio.cash
        logs['portfolio capital used'] = algo.portfolio.capital_used
        logs['portfolio positions exposure'] = algo.portfolio.positions_exposure
        logs['portfolio positions value'] = algo.portfolio.positions_value
        logs['number of orders'] = len(algo.blotter.orders)
        logs['number of open orders'] = len(algo.blotter.open_orders)
        logs['number of open positions'] = len(algo.portfolio.positions)
        
        # add recorded variables from `zipline.algorithm.record` method
        for name, value in algo.recorded_vars.items():
            logs[name] = value
        
        # add any extras passed in through `other_logs` dictionary
        for name, value in other_logs.items():
            logs[name] = value
        
        self.log_dict(epoch, logs, model_summaries)

