import torch
import torch.nn.functional as F
from .policy import ActorNetwork, CriticsNetwork
from numpy import log

class SAC(object):
    def __init__(self, 
                 obs_dim,
                 action_dim,
                 config,
                 device):

        self.config = config
        self.device = device
        self.gamma = config.gamma
        self.mini_batch_size = config.mini_batch_size

        self.actor = ActorNetwork(obs_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.learning_rate)
        
        self.critics = CriticsNetwork(obs_dim, action_dim, config.hidden_size)
        self.critics_target = CriticsNetwork(obs_dim, action_dim, config.hidden_size)
        self.critics_target.load_state_dict(self.critics.state_dict())
        for p in self.critics_target.parameters():
            p.requires_grad = False
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), config.learning_rate)

        self.log_alpha = torch.tensor(log(config.temp_init_value)).to(device)
        if not config.fixed_temperature:
            self.log_alpha.requires_grad = True
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], config.learning_rate)
        self.entropy_target = -action_dim

        self.actor.to(device)
        self.critics.to(device)
        self.critics_target.to(device)
        self.log_alpha.to(device)

    def act(self, obs, deterministic=False):
        action, _ = self.actor(obs, deterministic)
        return action

    def update(self, 
               replay_buffer, 
               iteration
               ):

        #obs, action, reward, next_obs, not_terminal
        sample = replay_buffer.sample(self.mini_batch_size)
        
        q_loss = self._update_critics(*sample)

        actor_loss, alpha_loss  = self._update_actor_alpha(sample[0])

        if iteration % self.config.critic_target_update==0:
            self._update_target_network()

        return actor_loss, q_loss, alpha_loss
    
    def _update_critics(self, 
                        obs, 
                        action, 
                        reward, 
                        next_obs,
                        not_terminal
                        ):

        with torch.no_grad():
            next_action, next_action_logit = self.actor(next_obs)
            q1, q2 = self.critics(next_obs, next_action)
            V_target = torch.min(q1,q2) - self.alpha * next_action_logit

            Q_target = reward + self.gamma * not_terminal * V_target

        q1_pred, q2_pred = self.critics(obs, action)

        loss = F.mse_loss(q1_pred, Q_target) + F.mse_loss(q2_pred, Q_target)
        
        self.critics_optimizer.zero_grad()
        loss.backward()
        self.critics_optimizer.step()
        
        return loss.item()

    def _update_actor_alpha(self, obs):
        
        """
        """

        action, action_logit = self.actor(obs)

        q1, q2 = self.critics(obs, action)
        q_pred = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * action_logit - q_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ######

        if not self.config.fixed_temperature:
            alpha_loss = (self.alpha * (-action_logit - self.entropy_target).detach()).mean()
    
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor([0.0])

        return actor_loss.item(), alpha_loss.item()


    def _update_target_network(self):

        critic_params = self.critics.parameters()
        target_params = self.critics_target.parameters()
        for c_params, t_params in zip(critic_params, target_params):
            t_params.data.copy_(self.config.tau * c_params.data + \
                               (1 - self.config.tau) * t_params.data)

    @property
    def alpha(self):
        return self.log_alpha.exp()
