import os
import gym
import random
import numpy as np
import pybullet as p
from baseControlEnv import BaseControlEnv

class SoloGaitMBEnv(BaseControlEnv):
    def __init__(self, config):
        self.T_gait = config.get('T_gait', 0.32)
        config['rl_dt'] = self.T_gait
        super(SoloGaitMBEnv, self).__init__(config)

        self.num_actions = int(4 * self.T_gait / self.dt / self.k_mpc)# 64
        self.action_space = gym.spaces.MultiBinary(self.num_actions) # No noop action

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 4 rel foot pose, 6 vel_ref, 10 gait seq = 62
        high = np.inf * np.ones([52])#62])       
        self.observation_space = gym.spaces.Box(-high, high)

    def set_new_gait(self, sequences):
        #fill gait pattern in teh 5 columns version
        gait = np.zeros((20,5))
        gait[:16,0] = 1
        gait[:16, 1:5] = sequences.reshape((-1,4))
        self.controller.planner.Cplanner.set_gait(gait)

    def get_observation(self):

        self.robot.UpdateMeasurment()
        qu = np.array([self.robot.baseState[0],
            p.getEulerFromQuaternion(self.robot.baseOrientation)]).flatten()[2:]

        qu_dot = np.array(self.get_base_vel()).flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes

        pfeet = self.get_feet_positions().flatten()

        #executed_gait = self.get_past_gait()[:2].flatten()

        history = ...

        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, self.vel_ref.flatten()])
