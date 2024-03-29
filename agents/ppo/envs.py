from multiprocessing import Process, Pipe
import torch
import numpy as np
from gym.core import Env


class NDProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

def make_vec_envs(config, num_envs, env_constructor, gamma=0.99, device=torch.device('cpu:0'), training=True):
    """construct env
    """
    def thunk_():
        return env_constructor(config)
    if num_envs >1:
        envs = VecEnvWrapper([thunk_ for _ in range(num_envs)])
    else:
        envs = DummyEnvWrapper([thunk_ for _ in range(num_envs)])


    from soloRL.agents.running_mean_std import VecNormalize
    envs = VecNormalize(envs, ob=False, ret=False, clipob=100, cliprew=100, gamma=gamma)
    if not training:
        envs.eval() 

    return PyTorchEnvWrapper(envs, device)

def simple_worker(remote, env):
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
            elif cmd == 'get_torques':
                remote.send(env.robot.tau_ff)
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'close':
                env.close()
                remote.send(True)
            elif cmd == 'reset_vel':
                env.reset_vel_ref(data)
                #remote.send(True)
            elif cmd == 'increment_curriculum':
                env.increment_curriculum()
                #remote.send(True)
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
            self.ps.append(NDProcess(target=simple_worker, args=(p, envs_fn[n])))
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
    
    def get_torques(self):
        for remote in self.remotes:
            remote.send(('get_torques', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_spaces(self):
        self.remotes[0].send(('get_spaces', None))
        return self.remotes[0].recv()

    def reset_vel(self, vel):
        for remote, v in zip(self.remotes, vel):
            remote.send(('reset_vel', v))
        #return self.remotes[0].recv()

    def increment_curriculum(self):
        for remote in self.remotes:
            remote.send(('increment_curriculum', None))

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

class DummyEnvWrapper(Env):
    def __init__(self, envs_fn):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.nenvs = 1

        self.envs = [envs_fn[0]()]
        self._observation_space = self.envs[0].observation_space
        self._action_space = self.envs[0].action_space
    
    def step(self, actions):
        o, r, d, info = self.envs[0].step(actions[0])
        return np.expand_dims(o,0).astype(np.float32), \
               np.expand_dims(r,0).astype(np.float32), \
               np.array((d,)),info

    def reset(self):
        return np.expand_dims(self.envs[0].reset(),0).astype(np.float32)

    def reset_vel(self, vel):
        self.envs[0].reset_vel_ref(vel)

    def close(self):
        self.envs[0].close()

    def __len__(self):
        return 1

    @property 
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

class PyTorchEnvWrapper(Env):
    def __init__(self, envs, device):
        self.envs = envs
        self.nenvs = len(self.envs)
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

    def get_torques(self):
        tau = self.envs.get_torques()
        return torch.from_numpy(tau).to(self.device).float()

    def increment_curriculum(self):
        self.envs.increment_curriculum()

    @property 
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space
