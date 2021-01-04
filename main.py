#from solo import SoloBase
#from baseEnv import SoloBaseEnv
from soloGaitEnv import SoloGaitEnv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser([])
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--episode-length', type=int, default=200)
    parser.add_argument('--mode', type=str, default='headless')
    parser.add_argument('--model-urdf', type=str, default='./solo_description/robots/solo.urdf')
    args = parser.parse_args()

    if args.config_file is not None:
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config['mode'] = args.mode
    else:
        config = vars(args)

    env = SoloGaitEnv(config)
    import pudb; pudb.set_trace()
