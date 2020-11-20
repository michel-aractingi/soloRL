"""
General training script that would start the run-scripts of the specified algorithm
"""
import os
import torch
import time
import numpy as np
from .sac import SAC
from .envs import get_envs
from .buffer import ReplayBuffer
from collections import deque, defaultdict
from soloRL.agents import utils 


def train(args, config, env_constructor, writer=None):    

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu:0')

    #create environment
    train_envs = get_envs(config, args.num_agents, env_constructor, device)

    #create actor-critic model
    obs_dim = train_envs.observation_space.shape[0]
    action_dim = train_envs.action_space.shape[0]

    #initialize training algorithm
    rl_agent = SAC(obs_dim, action_dim, args, device)
    
    #initialize episodic buffer
    replay_buffer = ReplayBuffer(args.max_replay_size, obs_dim, action_dim, device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True    

    num_updates = args.num_training_steps // args.num_agents

    # reset env and buffer
    observations = train_envs.reset()

    # seed the replay buffer with random transitions
    for _ in range(args.num_seed_steps, args.num_agents):
        actions = torch.randn(args.num_agents, action_dim)
        next_obs, rewards, dones, _ = train_envs.step(actions)
        not_terminal_mask = torch.Tensor([[0.0] if done else [1.0] for done in dones])
        replay_buffer.append(zip(observations, actions, rewards, next_obs, not_terminal_mask))

    ###########start training loop
    episode_length = deque(maxlen=30)
    episode_rewards = deque(maxlen=30)
    episode_success = deque(maxlen=32)
    episode_rewards_dict = defaultdict(lambda: deque(maxlen=32))

    print(observations.shape)
    step = 0
    starttime = time.time()
    
    for i in range(0, args.num_training_steps, args.num_agents):

        # Sample actions
        with torch.no_grad():
            actions = rl_agent.act(observations)
     
        # Obser reward and next obs
        next_obs, rewards, dones, infos = train_envs.step(actions)

        for info,done in zip(infos,dones):
            if done.item():
                episode_rewards.append(info['episode_reward'])
                episode_length.append(info['episode_length'])
                episode_success.append(info['success'])
                for k in info.keys():
                    if 'dr/' in k.lower():
                        episode_rewards_dict[k].append(info[k])

        # Append to storage 
        replay_buffer.append(zip(observations, actions, rewards, next_obs, 1 - dones))

        # perform update step
        ac_loss, q_loss, alpha_loss = rl_agent.update(replay_buffer, i)

        #check for visualization interval
        if (step + 1) % args.log_interval==0 and args.logdir is not None:
            fulltime = time.time() - starttime
            #tensor board
            utils.log(writer, ac_loss, 'Loss/action', i)
            utils.log(writer, q_loss, 'Loss/Qval', i)
            utils.log(writer, alpha_loss, 'alpha/loss', i)
            utils.log(writer, rl_agent.alpha.item(), 'alpha/value', i)
            utils.log(writer, episode_rewards, 'Episode/reward', i)
            utils.log(writer, episode_length, 'Episode/length', i)
            utils.log(writer, episode_success, 'Episode/success_mean', i)
            utils.log(writer, episode_rewards_dict, '', i)
            
            print("Num env frames {}, FPS {}, Last {} training episodes: mean/max reward {:.2f}/{:.2f}, critics loss {:.2f}, rl loss {:.2f}".format(
                i,
                int(i / (time.time() - starttime)),
                len(episode_rewards),
                np.mean(episode_rewards), np.max(episode_rewards),
                q_loss, ac_loss))

        #check for checkpoint interval
        if (step + 1) % args.save_interval==0 and args.logdir is not None:
            #add optimizer or lr if decay is used
            checkpoint = {'update': i, 'state_dict': rl_agent.actor.state_dict()}
            torch.save(checkpoint, os.path.join(args.logdir,'ckpt_{}.pth'.format(i)))

        step += 1

    if args.logdir is not None:
        checkpoint = {'update': num_updates, 'state_dict': rl_agent.actor.state_dict()}

        torch.save(checkpoint, os.path.join(args.logdir, 'ckpt_{}.pth'.format('final')))
    train_envs.close()
