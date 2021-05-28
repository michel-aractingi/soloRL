import copy
import glob
import os
import time
from collections import deque, defaultdict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from soloRL.agents import utils
from .envs import make_vec_envs
from .policy import Policy
from .storage import OPBuffer
from .ppo import PPO


def train(args, config, env_constructor, writer=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(config, args.num_agents, env_constructor, args.gamma, device)

    if envs.action_space.__class__.__name__ == 'Discrete':
        action_dim = envs.action_space.n
        storage_dim = 1
    else:
        action_dim = envs.action_space.shape[0]
        storage_dim = action_dim

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, {'hidden_size':args.hidden_size})
    actor_critic.to(device)

    print(actor_critic)

    agent = PPO(
                 actor_critic,
                 args.clip_param,
                 args.ppo_epoch,
                 args.mini_batch_size,
                 args.value_loss_coef,
                 args.entropy_coef,
                 lr=args.lr,
                 l2_coef=args.l2_coef,
                 max_grad_norm=args.max_grad_norm)

    ep_buffer = OPBuffer(args.num_steps, args.num_agents,
                              envs.observation_space.shape, storage_dim, device)

    obs = envs.reset()
    ep_buffer.obs[0].copy_(obs)

    episode_rewards = deque(maxlen=32)
    episode_length = deque(maxlen=32)
    episode_success = deque(maxlen=32)
    max_vel = 0.0; min_f = 0.0; max_f = 0.0
    episode_rewards_dict = defaultdict(lambda: deque(maxlen=32))

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_agents

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(ep_buffer.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info, d in zip(infos, done):
                if d.item():
                    episode_rewards.append(info['episode_reward'])
                    episode_length.append(info['episode_length'])
                    episode_success.append(info['success'])
                    max_vel = max(max_vel, info['max_velocity'])
                    min_f = max(min_f, info['min_force'])
                    max_f = max(max_f, info['max_force'])
                    for k in info.keys():
                        if 'dr/' in k.lower():
                            episode_rewards_dict[k].append(info[k])

            masks = (1 - done).unsqueeze(-1)
            ep_buffer.append(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(ep_buffer.obs[-1]).detach()

        ep_buffer.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.tau)

        value_loss, action_loss, dist_entropy = agent.update(ep_buffer)

        ep_buffer.reset()

        # Increment linear curriculum schedule 
        if args.curriculum_schedule and (j + 1) % args.curriculum_schedule == 0:
            envs.increment_curriculum()

        total_num_steps = (j + 1) * args.num_agents * args.num_steps
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.logdir is not None:

            torch.save({
                'update': j, 'state_dict': actor_critic.state_dict(),
                'ob_rms': getattr(envs.envs, 'ob_rms', None)}, 
                os.path.join(args.logdir, "solo_{}.pt".format(total_num_steps)))
            torch.save({
                'update': j, 'state_dict': actor_critic.state_dict(),
                'ob_rms': getattr(envs.envs, 'ob_rms', None)}, 
                os.path.join(args.logdir, "solo.pt"))

        if j % args.log_interval == 0:
            end = time.time()
            print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n mean/max length {:.0f}/{:0f}  mean success {:0f} \n entropy {:.1f} value loss {:.1f} action loss {:.1f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_length), np.max(episode_length), np.mean(episode_success), dist_entropy, value_loss,
                        action_loss))

            if args.logdir is not None:    
                utils.log(writer, value_loss, 'Loss/value', total_num_steps)
                utils.log(writer, action_loss, 'Loss/action', total_num_steps)
                utils.log(writer, dist_entropy, 'Loss/entropy', total_num_steps)
                utils.log(writer, episode_rewards, 'Episode/reward', total_num_steps)
                utils.log(writer, episode_success, 'Episode/success_mean', total_num_steps)
                utils.log(writer, episode_length, 'Episode/length', total_num_steps)
                utils.log(writer, max_vel, 'Curriculum/max_vel', total_num_steps)
                utils.log(writer, max_f, 'Curriculum/max_force', total_num_steps)
                utils.log(writer, min_f, 'Curriculum/min_force', total_num_steps)
                utils.log(writer, episode_rewards_dict, '', total_num_steps)


    torch.save({ 
        'update': j, 'state_dict': actor_critic.state_dict(),
        'ob_rms': getattr(envs.envs, 'ob_rms', None)}, 
         os.path.join(args.logdir, "solo.pt"))

    import sys; sys.exit()
