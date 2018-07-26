# -*- coding:utf-8 -*-
from agents.agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
import os


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, b_dim, rnn_layers=1, dp=0.2):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.rnn_layers = rnn_layers
        self.gru = nn.GRU(self.s_dim, 128, self.rnn_layers, batch_first=True)
        self.fc_s_1 = nn.Linear(128, 128)
        self.fc_s_2 = nn.Linear(128, 64)
        self.fc_s_out = nn.Linear(64, 1)
        self.fc_pg_1 = nn.Linear(128, 128)
        self.fc_pg_2 = nn.Linear(128, 64)
        self.fc_pg_out = nn.Linear(64, self.a_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dp)
        self.softmax = nn.Softmax(dim=-1)
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, 128, dtype=torch.float32)
    
    def forward(self, state, hidden=None, train=False):
        state, h = self.gru(state, hidden)
        if train:
            state = self.dropout(state)
        sn_out = self.relu(self.fc_s_1(state))
        sn_out = self.relu(self.fc_s_2(sn_out))
        sn_out = self.fc_s_out(sn_out)
        
        pn_out = self.relu(self.fc_pg_1(state))
        pn_out = self.relu(self.fc_pg_2(pn_out))
        pn_out = self.softmax(self.fc_pg_out(pn_out))
        return pn_out, sn_out, h.data


class RPGAgent(Agent):
    def __init__(self, s_dim, a_dim, b_dim, batch_length=64, learning_rate=1e-3, rnn_layers=1):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.pointer = 0
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        
        self.train_hidden = None
        self.trade_hidden = None
        self.actor = Actor(s_dim=self.s_dim, a_dim=self.a_dim, b_dim=self.b_dim, rnn_layers=rnn_layers)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
    
    def _trade(self, state, train=False):
        with torch.no_grad():
            a, _, self.trade_hidden = self.actor(state[:, None, :], self.trade_hidden, train=False)
        if train:
            return torch.multinomial(a[:, 0, :], 1)
        else:
            return a[:, 0, :].argmax(dim=1)
    
    def trade(self, state):
        state_ = torch.tensor(state)
        action = self._trade(state_).numpy()
        return action
    
    def train(self):
        self.optimizer.zero_grad()
        s = torch.stack(self.s_buffer).t()
        s_next = torch.stack(self.s_next_buffer).t()
        r = torch.stack(self.r_buffer).t()
        a = torch.stack(self.a_buffer).t()
        a_hat, s_next_hat, self.train_hidden = self.actor(s, self.train_hidden, train=True)
        mse_loss = torch.nn.functional.mse_loss(s_next_hat, s_next)
        nll = -torch.log(a_hat.gather(2, a))
        pg_loss = (nll * r).mean()
        loss = mse_loss + pg_loss
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def reset_model(self):
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        self.trade_hidden = None
        self.train_hidden = None
        self.pointer = 0
    
    def save_transition(self, state, action, reward, next_state):
        if self.pointer < self.batch_length:
            self.s_buffer.append(torch.tensor(state))
            self.a_buffer.append(torch.tensor(action))
            self.r_buffer.append(torch.tensor(reward[:, None], dtype=torch.float32))
            self.s_next_buffer.append(torch.tensor(next_state))
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.a_buffer.pop(0)
            self.r_buffer.pop(0)
            self.s_next_buffer.pop(0)
            self.s_buffer.append(torch.tensor(state))
            self.a_buffer.append(torch.tensor(action))
            self.r_buffer.append(torch.tensor(reward[:, None], dtype=torch.float32))
            self.s_next_buffer.append(torch.tensor(next_state))
    
    def load_model(self, model_path='./RPG_Torch'):
        self.actor = torch.load(model_path + '/model.pkl')
    
    def save_model(self, model_path='./RPG_Torch'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor, model_path + '/model.pkl')
