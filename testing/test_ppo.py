import os
import yaml
import torch
import argparse
from soloRL.agents.ppo.policy import Policy
from soloRL.agents.ppo.envs import make_vec_envs
import time

import numpy as np

import re, glob

def str2bool(x):
    if x is None: return
    if x.lower() in ('false','f','n','0'):
        return False
    elif x.lower() in ('true','t','y','1'):
        return True
    else:
        raise NotImplementedError

if __name__=='__main__':
    parser = argparse.ArgumentParser([])
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--mode', type=str, default='gui')
    parser.add_argument('--env-name', type=str, default='gait')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--store-action-histogram', action='store_true', default=False)
    parser.add_argument('--linear-velocity', action='store_true', default=False)
    parser.add_argument('--episode-length', type=int, default=None)
    parser.add_argument('--vel-switch', type=int, default=None)
    parser.add_argument('--reactive-update', type=str2bool, default=None)
    parser.add_argument('--add-external-force', type=str2bool, default=None)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if 'robot' in config.keys():
        config['robot']['enable_pyb_GUI'] = args.mode
    else:
        config['mode'] = args.mode

    if args.episode_length is not None: config['episode_length'] = args.episode_length
    if args.vel_switch is not None: config['vel_switch'] = args.vel_switch
    if args.reactive_update is not None: config['reactive_update'] = args.reactive_update
    if args.add_external_force is not None: config['add_external_force'] = args.add_external_force
    if args.task is not None:
        config['task'] = args.task

    if args.env_name == 'base':
        from soloRL.baseEnv import SoloBaseEnv as env_constructor
    elif args.env_name == 'gait':
        from soloRL.soloGaitEnv import SoloGaitEnv as env_constructor
    elif args.env_name == 'contact':
        from soloRL.soloGaitEnvContact import SoloGaitEnvContact as env_constructor
    elif args.env_name == 'gaitperiod':
        from soloRL.soloGaitPeriodEnv import SoloGaitPeriodEnv as env_constructor
    elif args.env_name == 'timing':
        from soloRL.soloTimingsEnv import SoloTimingsEnv as env_constructor
    elif args.env_name == 'timing12':
        from soloRL.soloTimingsEnv12 import SoloTimingsEnv12 as env_constructor
    elif args.env_name == 'timingdelta12':
        from soloRL.soloTimingsDeltaEnv12 import SoloTimingsDeltaEnv12 as env_constructor
    elif args.env_name == 'timingdeltamd':
        from soloRL.soloTimingsDeltaEnvMD import SoloTimingsDeltaEnvMD as env_constructor
    elif args.env_name == 'timingoneleg':
        from soloRL.soloTimingsOneLegEnv import SoloTimingsOneLegEnv as env_constructor
    elif args.env_name == 'timingoneleg4':
        from soloRL.soloTimingsOneLegEnv4 import SoloTimingsOneLegEnv4 as env_constructor
    else:
        raise NotImplementedError('Error Env {} not found!'.format(args.env_name))

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

    policy = Policy(env.observation_space.shape, env.action_space ,None, {'hidden_size':512})
    print(policy)
    policy.load_state_dict(ckpt['state_dict'])
    policy.eval()

    if args.store_action_histogram:
        a_hist = [0 for _ in range(env.action_space.n)]
        a_list = []
        
    if args.linear_velocity:
        vel = np.zeros((1,6))
    ep_len = []
    ep_success = []
    ep_R = []
    obs = env.reset()
    if args.linear_velocity:
        vel = np.zeros((1,6))
        env.envs.venv.reset_vel(vel)
    trqs = []
    vels = []
    episode = 0
    energy = []

    while episode < args.num_runs:
        with torch.no_grad():
            _, action, _ = policy.act(obs, deterministic=False)

        #action[0][0] = 0
        obs, reward, done, infos = env.step(action)

        if args.store_action_histogram:
            a_hist[int(action.item())] += 1
            a_list.append(int(action.item()))

        if infos['episode_length'] % 20==0:
            vels.append(infos['dr/body_velocity'])
            trqs.append(infos['dr/Torque_pen'])
            energy.append(infos['dr/Energy_pen'])
            if args.linear_velocity:
                vel[0][0] += 0.1
                env.envs.venv.reset_vel(vel)

        #time.sleep(0.05)

        if done:
            for i in reversed(range(1,len(energy))):
                energy[i] = energy[i] - energy[i-1]
                vels[i] = vels[i] - vels[i-1]
                trqs[i] = trqs[i] - trqs[i-1]
            #energy = np.array(energy)/20
            #vels = np.array(vels)/20
            #trqs = np.array(trqs)/20
            #import pudb; pudb.set_trace()
            print(episode, infos['episode_length'])
            episode += 1
            obs = env.reset()
            ep_len.append(infos['episode_length'])
            ep_R.append(infos['episode_reward'])
            ep_success.append(infos['success'])
    print('mean length {} mean reward {} mean success {}'.format(sum(ep_len)/args.num_runs,sum(ep_R)/args.num_runs, sum(ep_success)/args.num_runs))
    if args.store_action_histogram:
        print(a_hist)
    import pudb; pudb.set_trace()
    import sys; sys.exit()
