import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.config.config import settings


def init_hidden_layers(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # self.bn_1 = nn.BatchNorm1d(state_size)
        self.lin_1 = nn.Linear(state_size, settings.actor_size_1)
        self.bn_2 = nn.BatchNorm1d(settings.actor_size_1)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(settings.actor_size_1, settings.actor_size_2)
        self.bn_3 = nn.BatchNorm1d(settings.actor_size_2)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(settings.actor_size_2, action_size)
        self.tanh = nn.Tanh()

        self.initialize_params()

    def forward(self, x):
        # x = self.bn_1(x)
        x = torch.Tensor(x)
        x = self.lin_1(x)
        x = self.bn_2(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        x = self.bn_3(x)
        x = self.relu_2(x)
        x = self.lin_3(x)
        x = self.tanh(x)
        return x

    def initialize_params(self):
        self.lin_1.weight.data.uniform_(*init_hidden_layers(self.lin_1))
        self.lin_2.weight.data.uniform_(*init_hidden_layers(self.lin_2))
        self.lin_3.weight.data.uniform_(-3e-3, 3e-3)


class CriticNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # self.bn_1 = nn.BatchNorm1d(state_size)
        self.lin_1 = nn.Linear(state_size, settings.critic_size_1)
        self.bn_2 = nn.BatchNorm1d(settings.critic_size_1)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(settings.critic_size_1 + action_size, settings.critic_size_2)
        self.relu_2 = nn.ReLU()
        self.bn_3 = nn.BatchNorm1d(settings.critic_size_2)
        self.lin_3 = nn.Linear(settings.critic_size_2, 1)

        self.initialize_params()

    def forward(self, state, action):

        # concat = self.bn_1(state)
        state = torch.Tensor(state)
        action = torch.Tensor(action)

        concat = self.lin_1(state)
        concat = self.bn_2(concat)
        concat = self.relu_1(concat)
        concat = torch.cat([concat, action], dim=1)
        concat = self.lin_2(concat)
        concat = self.bn_3(concat)
        concat = self.relu_2(concat)
        concat = self.lin_3(concat)

        return concat

    def initialize_params(self):
        self.lin_1.weight.data.uniform_(*init_hidden_layers(self.lin_1))
        self.lin_2.weight.data.uniform_(*init_hidden_layers(self.lin_2))
        self.lin_3.weight.data.uniform_(-3e-3, 3e-3)
