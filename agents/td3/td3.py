import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Code adapted from https://github.com/sfujim/TD3
class TD3(object):
    def __init__(
    	self,
    	obs_dim,
    	action_dim,
    	gamma=0.99,
    	tau=0.005,
    	policy_noise=0.2,
    	noise_clip=0.5,
    	policy_freq=2):
    
    	self.actor = Actor(obs_dim, action_dim).to(device)
    	self.actor_target = copy.deepcopy(self.actor)
    	self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
    
    	self.critic = Critic(obs_dim, action_dim).to(device)
    	self.critic_target = copy.deepcopy(self.critic)
    	self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    
    	self.gamma = gamma
    	self.tau = tau
    	self.policy_noise = policy_noise
    	self.noise_clip = noise_clip
    	self.policy_freq = policy_freq
    
    def select_action(self, obs):
    	return self.actor(obs)
    
    def train(self, replay_buffer, step, batch_size=100):
        actor_loss = None
    
        # Sample replay buffer 
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
            	torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_obs) + noise
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if step % self.policy_freq == 0:
    
            # Compute actor losse
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        losses = critic_loss.item(), actor_loss.item() if actor_loss is not None else actor_loss

        return losses

    def save(self, filename):
    	torch.save(self.critic.obs_dict(), filename + "_critic")
    	torch.save(self.critic_optimizer.obs_dict(), filename + "_critic_optimizer")
    	
    	torch.save(self.actor.obs_dict(), filename + "_actor")
    	torch.save(self.actor_optimizer.obs_dict(), filename + "_actor_optimizer")
    
    
    def load(self, filename):
    	self.critic.load_obs_dict(torch.load(filename + "_critic"))
    	self.critic_optimizer.load_obs_dict(torch.load(filename + "_critic_optimizer"))
    	self.critic_target = copy.deepcopy(self.critic)
    
    	self.actor.load_obs_dict(torch.load(filename + "_actor"))
    	self.actor_optimizer.load_obs_dict(torch.load(filename + "_actor_optimizer"))
    	self.actor_target = copy.deepcopy(self.actor)
    	
