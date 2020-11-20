import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -2

def init_linear(layer):
    nn.init.orthogonal_(layer.weight.data)
    nn.init.constant_(layer.bias.data, 0.0)

class DiagGaussianHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.mean = nn.Linear(input_size, output_size)
        self.logstd = nn.Parameter(torch.zeros(1, output_size))
        self.train()

    def forward(self, x):
        mean = self.mean(x)
        logstd = self.logstd.expand(mean.size())

        return Normal(mean, logstd.exp())

class ActorNetwork(nn.Module):
    def __init__(self, 
                 obs_size,
                 action_size, 
                 hidden_size=64):

        super().__init__()
        self.net = nn.Sequential(
                     nn.Linear(obs_size, hidden_size),
                     nn.Tanh(),
                     nn.Linear(hidden_size, hidden_size),
                     nn.Tanh())

        self.mean = nn.Linear(hidden_size, action_size)
        self.logstd = nn.Linear(hidden_size, action_size)

        self.train()

    def forward(self, obs, deterministic=False):
        x = self.net(obs)

        mu = self.mean(x)
        logstd = self.logstd(x)

        pi = Normal(mu, logstd.exp())

        if not deterministic:
            action = pi.rsample()
        else:
            action = mu

        logprobs = (pi.log_prob(action) - torch.log(1 - action.tanh().pow(2))).sum(-1)

        action.tanh_()

        return action, logprobs.unsqueeze(-1)

class CriticsNetwork(nn.Module):
    def __init__(self, 
                 obs_size, 
                 action_size, 
                 hidden_size=64): 

        super().__init__()
        
        self.Q1 = nn.Sequential(
                    nn.Linear(obs_size + action_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1))
    
        self.Q2 = nn.Sequential(
                    nn.Linear(obs_size + action_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1))
    
        self.train()

    def forward(self, obs, action):
        
        assert obs.shape[0] == action.shape[0]
        obs_actions = torch.cat([obs, action], dim=-1)

        q1 = self.Q1(obs_actions)
        q2 = self.Q2(obs_actions)

        return q1, q2

