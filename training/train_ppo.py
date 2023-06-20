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
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--env-name', type=str, default='base')

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--clip-param', type=float, default=0.1)
    parser.add_argument('--ppo-epoch', type=int, default=10)
    parser.add_argument('--mini-batch-size', type=int, default=32)
    #parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2-coef', type=float, default=0.0)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--clip-value-loss', action='store_true', default=False)
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False)
    parser.add_argument('--use-gae', action='store_true', default=False)

    parser.add_argument('--num-env-steps', type=int, default=1e6)
    parser.add_argument('--seed', type=int, default=2301)
    parser.add_argument('--curriculum-schedule', type=int, default=0)

    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--base-checkpoint', type=str, default=None)
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
    print(config)
    #if args.env_name in ('gait', 'contact'):
        #config['task'] = ''
    #elif args.task is not None:
        #config['task'] = args.task
    
    args.episode_length = config['episode_length']
    args.num_steps = args.episode_length
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    else:
        args.timestamp = datetime.now().strftime('%Y%m%d-') + args.timestamp
    
    if args.logdir is not None:
        task = args.task + '_' if args.task is not None else ''
        args.logdir = os.path.join(args.logdir, 'Solo' + args.env_name.capitalize() + '_' + task + args.timestamp)
        writer = SummaryWriter(args.logdir)
    else: 
        writer = None

    if args.env_name == 'base':
        from soloRL.baseEnv import SoloBaseEnv as env_constructor
    elif args.env_name == 'gait':
        from soloRL.soloGaitEnv import SoloGaitEnv as env_constructor
    elif args.env_name == 'contact':
        from soloRL.soloGaitEnvContact import SoloGaitEnvContact as env_constructor
    elif args.env_name == 'mbgait':
        from soloRL.soloGaitMBEnv import SoloGaitMBEnv as env_constructor
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

    ppo.train(args, config, env_constructor, writer)
