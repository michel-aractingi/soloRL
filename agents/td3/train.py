import numpy as np
import torch
import os
import time

from soloRL.agents import utils
from .td3 import TD3
from .envs import make_vec_envs
from .buffer import ReplayBuffer
from collections import deque, defaultdict


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
    	obs, done = eval_env.reset(), False
    	while not done:
    	    action = policy.select_action(np.array(obs))
    	    obs, reward, done, _ = eval_env.step(action)
    	    avg_reward += reward
    
    avg_reward /= eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(args, config, env_constructor, writer=None):
	
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(config, args.num_agents, env_constructor, args.gamma, device)
    
    # Set seeds
    obs_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0] 
    
    kwargs = {
    	"obs_dim": obs_dim,
    	"action_dim": action_dim,
    	"gamma": args.gamma,
    	"tau": args.tau,
    }
    
    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise 
    kwargs["noise_clip"] = args.noise_clip 
    kwargs["policy_freq"] = args.policy_freq

    policy = TD3(**kwargs)

    '''
    if args.load_model != "":
    	policy_file = file_name if args.load_model == "default" else args.load_model
    	policy.load(f"./models/{policy_file}")
    '''

    replay_buffer = ReplayBuffer(args.max_replay_size, 
                                 obs_dim, action_dim,
                                 device)
    
    # Evaluate untrained policy
    #evaluations = [eval_policy(policy, args.env, args.seed)]
    
    obs = envs.reset()
    episode_rewards = deque(maxlen=32)
    episode_length = deque(maxlen=32)
    episode_success = deque(maxlen=32)
    episode_rewards_dict = defaultdict(lambda: deque(maxlen=32))
    step = 0


    print("---------------------------------------")
    print(f"Policy: TD3, Env: SOLO, Seed: {args.seed}")
    print("---------------------------------------")
    

    starttime = time.time()
    for t in range(0, int(args.num_env_steps), args.num_agents):
        
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = torch.randn(args.num_agents, action_dim)
        else:
            a_noise = torch.FloatTensor(np.random.normal(0, args.expl_noise, 
                                                     size=(args.num_agents, action_dim))).to(device)
            with torch.no_grad():
                action = policy.select_action(obs) + a_noise        

        # Perform action
        next_obs, rewards, dones, infos = envs.step(action) 
        
        for info, d in zip(infos, dones):
            if d.item():
                episode_rewards.append(info['episode_reward'])
                episode_length.append(info['episode_length'])
                episode_success.append(info['success'])
                for k in info.keys():
                    if 'dr/' in k.lower():
                        episode_rewards_dict[k].append(info[k])

        replay_buffer.append(zip(obs, action, rewards, next_obs, 1 - dones))

        obs = next_obs
        
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            q_loss, ac_loss = policy.train(replay_buffer, step, args.batch_size)
        
            if step  % args.log_interval==0 and args.logdir is not None:
                fulltime = time.time() - starttime
                #tensor board
                utils.log(writer, ac_loss, 'Loss/action', t)
                utils.log(writer, q_loss, 'Loss/Qval', t)
                utils.log(writer, episode_rewards, 'Episode/reward', t)
                utils.log(writer, episode_length, 'Episode/length', t)
                utils.log(writer, episode_success, 'Episode/success_mean', t)
                utils.log(writer, episode_rewards_dict, '', t)
                
                print("Num env frames {}, FPS {}: mean/max reward {:.2f}/{:.2f}, critics loss {:.2f}, rl loss {:.2f}".format(
                    t, 
                    int(t / (time.time() - starttime)),
                    np.mean(episode_rewards), np.max(episode_rewards),
                    q_loss, ac_loss))

            #check for checkpoint interval
            if step % args.save_interval==0 and args.logdir is not None:
                #add optimizer or lr if decay is used
                checkpoint = {'update': t, 'state_dict': policy.actor.state_dict(),
                              'critic_state_dict': policy.critic.state_dict()}
                torch.save(checkpoint, os.path.join(args.logdir,'ckpt_{}.pth'.format(t)))

        step += 1

    if args.logdir is not None:
        checkpoint = {'update': args.num_env_steps, 'state_dict': policy.actor.state_dict(),
                      'critic_state_dict': policy.critic.state_dict()}

        torch.save(checkpoint, os.path.join(args.logdir, 'ckpt_{}.pth'.format('final')))
        """
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
    	    evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
        """
