import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from soloRL.agents.utils import init_layer

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()

        self.base = MLP(obs_shape[0])#, **base_kwargs)

        if action_space.__class__.__name__ == 'Discrete':
            self.pi_dist = CategoricalHead(self.base.output_size, action_space.n)
        elif action_space.__class__.__name__ == 'Box':
            self.pi_dist = DiagGaussian(self.base.output_size, action_space.shape[0])
        else:
            raise NotImplementedError

    def act(self, inputs,  deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.pi_dist(actor_features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
            
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.pi_dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLP, self).__init__()

        self.output_size = hidden_size

        self.features = nn.Sequential(
            init_layer(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_layer(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_layer(nn.Linear(hidden_size, 1)))

        self.train()

    def forward(self, x):

        x_features = self.features(x)
        value = self.critic(x)
        return value, x_features

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.mean = init_layer(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand(action_mean.size())

        return ModNormal(action_mean, action_logstd.exp())

class CategoricalHead(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CategoricalHead, self).__init__()
        self.linear = init_layer(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        return ModCategorical(logits=self.linear(x))

class ModNormal(Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class ModCategorical(Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

