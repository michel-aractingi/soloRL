#from solo import SoloBase
#from baseEnv import SoloBaseEnv
from soloGaitEnvContact import SoloGaitEnvContact
from soloGaitEnv import SoloGaitEnv
from soloGaitMBEnv import SoloGaitMBEnv
from soloGaitPeriodEnv import SoloGaitPeriodEnv
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

    env = SoloGaitPeriodEnv(config)
    import pudb; pudb.set_trace()



#Test Base line actions
"""
actions =...
runs = 3
e_trqs = np.zeros((runs,len(actions), 10))
e_vels = np.zeros((runs,len(actions), 10))

for j in range(runs):
    trqs = e_trqs[j]
    vels = e_vels[j]
    for ai, a in enumerate(actions):
        env.reset()
        ref_vel = np.zeros(6)
        env.reset_vel_ref(ref_vel)
        print('current_action ', a)
        for i in range(0,10):
            for t in range(20):
                _,_,done,infos = env.step(ai)
            print('step ', i)
            trqs[ai, i] = infos['dr/Torque_pen']/20 * int(not done)
            vels[ai,i] = infos['dr/body_velocity']/20 * int(not done)
            env.reset()
            ref_vel[0] += .1
            env.reset_vel_ref(ref_vel)
"""

