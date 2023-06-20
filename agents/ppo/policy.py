import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli


from soloRL.agents.utils import init_layer

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs={}):
        super(Policy, self).__init__()

        if len(obs_shape) == 1:
            self.base = MLP(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 2:
            self.base = TransformerBase(obs_shape, **base_kwargs)

        if base is not None:
            self.base.load_state_dict(base)

        if action_space.__class__.__name__ == 'Discrete':
            self.pi_dist = CategoricalHead(self.base.output_size, action_space.n)
        elif action_space.__class__.__name__ == 'Box':
            self.pi_dist = DiagGaussian(self.base.output_size, action_space.shape[0])
        elif action_space.__class__.__name__ == 'MultiBinary':
            self.pi_dist = BernoulliHead(self.base.output_size, action_space.n)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            self.pi_dist = MultiCategoricalHead(self.base.output_size, action_space.nvec)
        else:
            raise NotImplementedError

    def act(self, inputs,  deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.pi_dist(actor_features)

        if deterministic:
            action = dist.mode()
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

class TransformerBase(nn.Module):
    def __init__(self, in_shape, nhead=8, hidden_size=64):
        super().__init__()

        dropout = 0.0

        self.seq_len, self.d_model = in_shape

        self.transformer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                      nhead=nhead,
                                                      dropout=dropout)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout, max_len=self.seq_len)

        mask = (torch.triu(torch.ones(self.seq_len, self.seq_len)) == 1).transpose(0, 1)
        self.src_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        self.output_size = hidden_size
        transformer_out = in_shape[0] * in_shape[1]

        self.features = nn.Sequential(
            init_layer(nn.Linear(transformer_out, hidden_size)), nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_layer(nn.Linear(transformer_out, hidden_size)), nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_layer(nn.Linear(hidden_size, 1)))

        self.train()

    def forward(self, x):

        x_transformer = self.transformer(self.pos_encoder(x))#, src_mask=self.src_mask)
        x_flat = x_transformer.view((x.shape[0], -1))
        x_features = self.features(x_flat)
        value = self.critic(x_flat)
        return value, x_features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.squeeze()
        return self.dropout(x)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.mean = init_layer(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand(action_mean.size())

        return ModNormal(action_mean, action_logstd.exp())

class MultiCategoricalHead(nn.Module):
    def __init__(self, num_inputs, nvec):
        super(MultiCategoricalHead, self).__init__()
        from soloRL.agents.multi_categorical_distribution import MultiCategoricalDistribution
        self.base_dist = MultiCategoricalDistribution(nvec)        
        self.linear = nn.Linear(num_inputs, sum(nvec))

    def forward(self, x):
        return self.base_dist.proba_distribution(self.linear(x))

class BernoulliHead(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(BernoulliHead, self).__init__()
        self.linear = init_layer(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        return ModBernoulli(logits=self.linear(x))
     
class ModNormal(Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    #def entropy(self):
        #return super().entropy()

    def mode(self):
        return self.mean

class ModCategorical(Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class ModBernoulli(Bernoulli):
    def log_probs(self, actions):
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()
