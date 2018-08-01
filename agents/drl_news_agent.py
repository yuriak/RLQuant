# -*- coding:utf-8 -*-
from agents.agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class Actor(nn.Module):
    def __init__(self, s_dim, b_dim, n_dim, rnn_layers=1, dp=0.2):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.n_dim = n_dim
        self.rnn_layers = rnn_layers
        self.state_gru = nn.GRU(self.s_dim, 128, self.rnn_layers, batch_first=True)
        self.news_gru = nn.GRU(self.n_dim, 64, self.rnn_layers, batch_first=True)
        self.fc_policy_1 = nn.Linear(128, 128)
        self.fc_policy_2 = nn.Linear(128, 64)
        self.fc_policy_out = nn.Linear(64, 1)
        self.fc_cash_out = nn.Linear(64, 1)
        
        self.news_fc_1 = nn.Linear(64, 32)
        self.news_fc_out = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dp)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, state, news, state_hidden=None, news_hidden=None, train=False):
        state, hs = self.state_gru(state, state_hidden)
        news, hn = self.news_gru(news, news_hidden)
        if train:
            state = self.dropout(state)
            news = self.dropout(news)
        state = self.relu(self.fc_policy_1(state))
        state = self.relu(self.fc_policy_2(state))
        
        news = self.relu(self.news_fc_1(news))
        cash = self.fc_cash_out(state) * self.news_fc_out(news)
        
        action = self.sigmoid(self.fc_policy_out(state)).squeeze(-1).t()
        cash = self.sigmoid(cash).mean(dim=0)
        action = torch.cat(((1 - cash) * action, cash), dim=-1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-10)
        return action, hs.data, hn.data


class DRLAgent(Agent):
    def __init__(self, s_dim, b_dim, n_dim, batch_length=64, learning_rate=1e-3, rnn_layers=1):
        super().__init__()
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.n_dim = n_dim
        self.batch_length = batch_length
        self.pointer = 0
        self.s_buffer = []
        self.d_buffer = []
        self.n_buffer = []
        
        self.state_train_hidden = None
        self.news_train_hidden = None
        self.state_trade_hidden = None
        self.news_trade_hidden = None
        
        self.actor = Actor(s_dim=self.s_dim, b_dim=self.b_dim, n_dim=self.n_dim, rnn_layers=rnn_layers)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
    
    def _trade(self, state, news, train=False):
        with torch.no_grad():
            a, self.state_trade_hidden, self.news_trade_hidden = self.actor(state=state,
                                                                            news=news,
                                                                            state_hidden=self.state_trade_hidden,
                                                                            news_hidden=self.news_trade_hidden,
                                                                            train=train)
        return a
    
    def trade(self, state, news, train=False):
        state_ = torch.tensor(state[:, None, :], dtype=torch.float32)
        news_ = torch.tensor(news[None, None, :], dtype=torch.float32)
        action = self._trade(state=state_, news=news_, train=train)
        return action.numpy().flatten()
    
    def train(self):
        self.optimizer.zero_grad()
        s = torch.stack(self.s_buffer).t()
        d = torch.stack(self.d_buffer)
        n = torch.stack(self.n_buffer)[None, :, :]
        a_hat, self.state_train_hidden, self.news_train_hidden = self.actor(state=s,
                                                                            news=n,
                                                                            state_hidden=self.state_train_hidden,
                                                                            news_hidden=self.news_train_hidden,
                                                                            train=True)
        reward = -(a_hat[:, :-1] * d).mean()
        reward.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def reset_model(self):
        self.s_buffer = []
        self.d_buffer = []
        self.n_buffer = []
        self.state_trade_hidden = None
        self.state_train_hidden = None
        self.news_trade_hidden = None
        self.news_train_hidden = None
        self.pointer = 0
    
    def save_transition(self, state, news, diff):
        if self.pointer < self.batch_length:
            self.s_buffer.append(torch.tensor(state, dtype=torch.float32))
            self.d_buffer.append(torch.tensor(diff, dtype=torch.float32))
            self.n_buffer.append(torch.tensor(news, dtype=torch.float32))
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.d_buffer.pop(0)
            self.n_buffer.pop(0)
            self.s_buffer.append(torch.tensor(state, dtype=torch.float32))
            self.d_buffer.append(torch.tensor(diff, dtype=torch.float32))
            self.n_buffer.append(torch.tensor(news, dtype=torch.float32))
    
    def load_model(self, model_path='./DRL_Torch'):
        self.actor = torch.load(model_path + '/model.pkl')
    
    def save_model(self, model_path='./DRL_Torch'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor, model_path + '/model.pkl')
