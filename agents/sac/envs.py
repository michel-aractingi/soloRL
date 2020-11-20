from multiprocessing import Process, Pipe
import torch
import numpy as np
from gym.core import Env

def get_envs(config, num_envs, env_constructor, 
              device=torch.device('cpu:0'), rms_norm=False, training=True):
    """construct env
    """
    def thunk_():
        return env_constructor(config)
    envs = VecEnvWrapper([thunk_ for _ in range(num_envs)])

    #from soloRL.agents.running_mean_std import VecNormalize
    #envs = VecNormalize(envs, ob=rms_norm, ret=rms_norm, clipob=100, cliprew=100, gamma=0.99)
    #if not training:
    #    envs.eval() 

    return PyTorchEnvWrapper(envs, device)

def simple_worker(remote, env_wrapper):
    
    env = env()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, rw, done, info = env.step(data)
                if done: ob = env.reset()
                remote.send((ob, rw, done, info))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'get_observation':
                remote.send(env.get_observation())
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'close':
                env.close()
                remote.send(True)
            else:
                raise NotImplementedError

    except KeyboardInterrupt:
        print('VecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

class VecEnvWrapper(Env):
    """Similar to the SubprocVec env in baselines.
    """
    def __init__(self, envs_fn):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        self.nenvs = len(envs_fn)

        self.envs = []
        self.ps= []
        self.remotes=[]
        self.work_remotes = []
        for n in range(self.nenvs):
            r, p = Pipe()
            self.remotes.append(r)
            self.work_remotes.append(p)
            self.ps.append(Process(target=simple_worker, args=(p, envs_fn[n])))
            self.ps[-1].daemon = True
            self.ps[-1].start()
            #self.work_remotes[-1].close()

        self._observation_space, self._action_space = self.get_spaces()
    
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        o, r, d, info = zip(*[remote.recv() for remote in self.remotes])
        return np.stack(o), np.stack(r), np.stack(d), info

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_observation(self):
        for remote in self.remotes:
            remote.send(('get_observation', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_spaces(self):
        self.remotes[0].send(('get_spaces', None))
        return self.remotes[0].recv()

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            import sys; sys.exit() #TODO fix join
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

    @property 
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class PyTorchEnvWrapper(Env):
    def __init__(self, envs, device):
        self.envs = envs
        self.nenvs = self.envs.nenvs
        self.device = device

    def step(self, actions):
        ob, rw, done, info = self.envs.step(actions.cpu().numpy())

        ob = torch.from_numpy(ob).to(self.device).float()
        rw = torch.from_numpy(rw).to(self.device).float()
        done = torch.from_numpy(done).to(self.device).float()

        return ob, rw.unsqueeze(-1), done, info

    def reset(self):
        ob = self.envs.reset()
        return torch.from_numpy(ob).to(self.device).float()

    def close(self):
        self.envs.close()

    def get_observation(self):
        ob = self.envs.get_observation()
        return torch.from_numpy(ob).to(self.device).float()

    @property 
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space
        
