import os
import yaml
import torch
import argparse
from datetime import datetime 
from soloRL.agents import td3
from soloRL.baseEnv import SoloBaseEnv
from torch.utils.tensorboard import SummaryWriter

def get_td3_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='base')

    parser.add_argument("--seed", default=0, type=int)              
    parser.add_argument("--start-timesteps", default=25e3, type=int)
    parser.add_argument("--eval-freq", default=5e3, type=int)       
    parser.add_argument("--num-env-steps", default=1e6, type=int)   
    parser.add_argument("--expl-noise", default=0.1)                
    parser.add_argument("--batch-size", default=256, type=int)      
    parser.add_argument("--gamma", default=0.99)                 
    parser.add_argument("--tau", default=0.005)                     
    parser.add_argument("--policy-noise", default=0.2)              
    parser.add_argument("--noise-clip", default=0.5)                
    parser.add_argument("--policy-freq", default=2, type=int)       
    parser.add_argument("--load-model", default="")                 
    parser.add_argument('--max-replay-size', type=int, default=1000000)

    parser.add_argument('--num-agents', type=int, default=32)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=2000)

    parser.add_argument('--config-file', type=str, default='../configs/basic.yaml')
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args()

def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__=='__main__':
    args = get_td3_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    config = parse_config(args.config_file)
    if args.task is not None:
        config['task'] = args.task
    
    args.episode_length = config['episode_length']
    args.num_steps = args.episode_length
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    else:
        args.timestamp = datetime.now().strftime('%Y%m%d-') + args.timestamp
    
    if args.logdir is not None:
        task = args.task + '_' if args.task is not None else ''
        args.logdir = os.path.join(args.logdir, 'Solo_td3_' + args.env_name.capitalize() + '_' + task + args.timestamp)
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
    elif args.env_name == 'timingoneleg':
        from soloRL.soloTimingsOneLegEnv import SoloTimingsOneLegEnv as env_constructor
    elif args.env_name == 'timingoneleg4':
        from soloRL.soloTimingsOneLegEnv4 import SoloTimingsOneLegEnv4 as env_constructor
    else:
        raise NotImplementedError('Error Env {} not found!'.format(args.env_name))

    td3.train(args, config, env_constructor, writer)

