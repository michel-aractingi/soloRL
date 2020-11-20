import os
import yaml
import torch
import argparse
from datetime import datetime 
from soloRL.agents import sac
from soloRL.baseEnv import SoloBaseEnv
from torch.utils.tensorboard import SummaryWriter

def get_ppo_args():
    parser = argparse.ArgumentParser('SAC args')
    parser.add_argument('--num-agents', type=int, default=32)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--temp-init-value', type=int, default=0.1)
    parser.add_argument('--fixed-temperature', action='store_true', default=False)
    parser.add_argument('--mini-batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--critic-target-update', type=int, default=2)
    parser.add_argument('--num-seed-steps', type=int, default=5000)
    parser.add_argument('--max-replay-size', type=int, default=1000000)

    parser.add_argument('--num-training-steps', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=2301)

    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=2000)

    parser.add_argument('--config-file', type=str, default='../configs/basic.yaml')
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args()

def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__=='__main__':
    args = get_ppo_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    config = parse_config(args.config_file)
    if args.task is not None:
        config['task'] = args.task
    
    args.episode_length = config['episode_length']
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    else:
        args.timestamp = datetime.now().strftime('%Y%m%d-') + args.timestamp
    
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, config['task'], 'Solo_' + args.timestamp)
        writer = SummaryWriter(args.logdir)
    else: 
        writer = None

    sac.train(args, config, SoloBaseEnv, writer)
