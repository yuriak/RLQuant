# -*- coding:utf-8 -*-
from abc import abstractmethod


class Agent(object):
    def __init__(self):
        pass
    
    @abstractmethod
    def trade(self, state):
        pass
    
    def train(self):
        pass
    
    @abstractmethod
    def load_model(self, model_path):
        pass
    
    @abstractmethod
    def save_model(self, model_path):
        pass