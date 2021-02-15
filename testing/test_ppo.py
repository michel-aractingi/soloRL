import os
import yaml
import torch
import argparse
from soloRL.agents.ppo.policy import Policy
from soloRL.agents.ppo.envs import make_vec_envs
import time

import re, glob

if __name__=='__main__':
    parser = argparse.ArgumentParser([])
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--mode', type=str, default='gui')
    parser.add_argument('--env-name', type=str, default='gait')
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['mode'] = args.mode
    if args.task is not None:
        config['task'] = args.task

    if args.env_name == 'base':
        from soloRL.baseEnv import SoloBaseEnv
        env_constructor = SoloBaseEnv
    elif args.env_name == 'gait':
        from soloRL.soloGaitEnv import SoloGaitEnv
        env_constructor = SoloGaitEnv
    elif args.env_name == 'contact':
        from soloRL.soloGaitEnvContact import SoloGaitEnvContact
        env_constructor = SoloGaitEnvContact

    os.chdir(args.checkpoint_dir)
    """
    ckpts = glob.glob('ckpt_*')
    if 'ckpt_final.pth' in ckpts:
        filename = 'ckpt_final.pth'
    else:
        ckpts.sort(key=lambda n: int(re.sub('[^0-9]','',n)))
        filename = ckpts[-1]
    """
    filename = "solo.pt"
    ckpt = torch.load(filename)

    env = make_vec_envs(config, 1, env_constructor, training=False)
    if 'ob_rms' in ckpt.keys():
        env.envs.ob_rms = ckpt['ob_rms']

    policy = Policy(env.observation_space.shape, env.action_space)
    policy.load_state_dict(ckpt['state_dict'])
    policy.eval()
        
    ep_len = []
    ep_R = []
    obs = env.reset()
    episode = 0
    N = 10
    while episode < 10:
        with torch.no_grad():
            _, action, _ = policy.act(obs, deterministic=True)

        obs, reward, done, infos = env.step(action)
        
        if done:
            ep_len.append(infos[0]['episode_length'])
            ep_R.append(infos[0]['episode_reward'])
            print(episode)
            episode += 1
    print('mean length {} mean reward {}'.format(sum(ep_len)/10,sum(ep_R)/10))
    #import pudb; pudb.set_trace()
    import sys; sys.exit()
