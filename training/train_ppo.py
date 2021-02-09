import os
import yaml
import torch
import argparse
from datetime import datetime 
from soloRL.agents import ppo
from torch.utils.tensorboard import SummaryWriter

def get_ppo_args():
    parser = argparse.ArgumentParser('PPO args')
    parser.add_argument('--num-agents', type=int, default=32)
    parser.add_argument('--output-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--env-name', choices=['base', 'gait', 'contact'], default='base')

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--clip-param', type=float, default=0.1)
    parser.add_argument('--ppo-epoch', type=int, default=10)
    parser.add_argument('--mini-batch-size', type=int, default=32)
    #parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--clip-value-loss', action='store_true', default=False)
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False)
    parser.add_argument('--use-gae', action='store_true', default=False)

    parser.add_argument('--num-env-steps', type=int, default=1e6)
    parser.add_argument('--seed', type=int, default=2301)

    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=20)

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
    if args.env_name in ('gait', 'contact'):
        config['task'] = ''
    elif args.task is not None:
        config['task'] = args.task
    
    args.episode_length = config['episode_length']
    args.num_steps = args.episode_length
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    else:
        args.timestamp = datetime.now().strftime('%Y%m%d-') + args.timestamp
    
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, config['task'], 'Solo' + args.env_name.capitalize() + '_' + args.timestamp)
        writer = SummaryWriter(args.logdir)
    else: 
        writer = None

    if args.env_name == 'base':
        from soloRL.baseEnv import SoloBaseEnv as env_constructor
    elif args.env_name == 'gait':
        from soloRL.soloGaitEnv import SoloGaitEnv as env_constructor
    elif args.env_name == 'contact':
        from soloRL.soloGaitEnvContact import SoloGaitEnvContact as env_constructor

    ppo.train(args, config, env_constructor, writer)
