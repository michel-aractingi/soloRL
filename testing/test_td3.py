import yaml
import torch
import argparse
from soloRL.agents.td3.models import Actor, Critic
from soloRL.agents.td3.envs import make_vec_envs
from soloRL.baseEnv import SoloBaseEnv

import os
import time
import re, glob

if __name__=='__main__':
    parser = argparse.ArgumentParser([])
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--config-file', type=str, default='../configs/basic.yaml')
    parser.add_argument('--mode', type=str, default='gui')
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['mode'] = args.mode
    if args.task is not None:
        config['task'] = args.task

    os.chdir(args.checkpoint_dir)
    ckpts = glob.glob('ckpt_*')
    if 'ckpt_final.pth' in ckpts:
        filename = 'ckpt_final.pth'
    else:
        ckpts.sort(key=lambda n: int(re.sub('[^0-9]','',n)))
        filename = ckpts[-1]
    print(filename)
    ckpt = torch.load(filename)

    env = make_vec_envs(config, 1, SoloBaseEnv, training=False)
    if 'ob_rms' in ckpt.keys():
        env.envs.ob_rms = ckpt['ob_rms']

    policy = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    policy.load_state_dict(ckpt['state_dict'])
    policy.eval()
        
    obs = env.reset()
    while True:
        with torch.no_grad():
            action = policy(obs)
            print(action)

        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
        time.sleep(0.05)

    env.close()
