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
        
    obs = env.reset()
    while True:
        with torch.no_grad():
            _, action, _ = policy.act(obs, deterministic=True)

        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
        time.sleep(0.05)

    env.close()
