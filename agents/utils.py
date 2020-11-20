import os
import numpy as np
import torch.nn as nn

def init_logging(logdir):
    if logdir is None:
        return 

    savedir =  os.path.join(logdir,'checkpoints')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    return savedir

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
   """Decreases the learning rate linearly"""
   lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
   for param_group in optimizer.param_groups:
       param_group['lr'] = lr

def log(writer, values, name, step):
    if writer is None:
        return

    if np.isscalar(values):
        writer.add_scalar(name, values, step)

    elif isinstance(values, dict):
        for k in values.keys():
            value = values[k] if np.isscalar(values[k]) else np.mean(values[k])
            writer.add_scalar(name + k, np.mean(value), step)
    else:
        if values: # true if not empty
            writer.add_scalar(name+'/min', np.min(values), step)
            writer.add_scalar(name+'/max', np.max(values), step)
            writer.add_scalar(name+'/mean', np.mean(values), step)


######################################
# Running mean std normalization code
######################################
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class VecNormalize:
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):

        self.venv = venv
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self._ret_shape = (self.venv.nenvs,)
        if isinstance(self.action_space, list):
            self._ret_shape = (self.venv.nenvs, len(self.action_space))
        self.ret = np.zeros(self._ret_shape)
        self.gamma = gamma
        self.epsilon = epsilon

        self.training = True

    def step(self, actions):
        obs, rews, news, infos = self.venv.step(actions)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)

        if self.ret_rms:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.

        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self._ret_shape)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def eval(self):
        self.training = False

def init_layer(layer):
    """
    Layer initialization function according to init functions specified in the filler args.
    Args:
      layer: layer to initialize
    """
    nn.init.orthogonal_(layer.weight.data, gain=2**0.5)
    if layer.bias is not None:
        nn.init.constant_(layer.bias.data, 0.0)

    return layer
