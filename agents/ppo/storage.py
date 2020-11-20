import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class OPBuffer(object):
    def __init__(self, num_steps, num_agents, obs_shape, action_dim, device):

        self.obs = torch.zeros(num_steps + 1, num_agents, *obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_agents, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_agents, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_agents, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_agents, 1).to(device)

        self.actions = torch.zeros(num_steps, num_agents, action_dim).to(device)
        self.masks = torch.ones(num_steps + 1, num_agents, 1).to(device)

        self.num_samples = num_steps * num_agents
        self.num_steps = num_steps
        self.device = device
        self.step = 0

    def append(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def reset(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae=True,
                        gamma=0.99,
                        gae_lambda=0.95):

        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step +
                                           1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step +
                                                              1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def batch_generator(self, advantages, mini_batch_size):

        
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_samples)), mini_batch_size, drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch,\
                                 masks_batch, old_action_log_probs_batch, adv_targ
