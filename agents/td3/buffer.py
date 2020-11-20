import torch
from typing import List
from collections import namedtuple
from gym.spaces import Discrete
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

transition_tuple = namedtuple('Transition', 
                     ['state', 'action', 'reward', 'next_state', 'not_terminal'])

class ReplayBuffer(object):
    def __init__(self, max_size, observation_dim,  action_dim, device):
        
        self.device = device
        self._max_size = max_size
        
        self._observations = torch.zeros((max_size, observation_dim))
        self._actions  = torch.zeros((max_size, action_dim))
        self._rewards = torch.zeros((max_size, 1))
        self._next_observations = torch.zeros((max_size, observation_dim))
        self._not_terminal = torch.ones((max_size, 1))

        self._size = 0
        self._top = 0
        
    def append(self, transitions: List[tuple]):
        for trans in transitions:
            self._add_sample(transition_tuple(*trans))

    def _add_sample(self, transition: transition_tuple):

        self._observations[self._top] = transition.state
        self._actions[self._top] = transition.action
        self._rewards[self._top] = transition.reward
        self._next_observations[self._top] = transition.next_state
        self._not_terminal[self._top] = transition.not_terminal

        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size: 
            self._size += 1

    def sample(self, mini_batch_size):

       indices = torch.randint(low=0, high=self._size, size=(mini_batch_size,1)).squeeze()

       obs_batch = self._observations[indices].to(self.device)
       action_batch = self._actions[indices].to(self.device)
       reward_batch = self._rewards[indices].to(self.device)
       next_obs_batch = self._next_observations[indices].to(self.device)
       not_terminal_batch = self._not_terminal[indices].to(self.device)

       return obs_batch, action_batch, reward_batch, next_obs_batch, not_terminal_batch

