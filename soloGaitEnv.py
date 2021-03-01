import os
import gym
import random
import numpy as np
import pybullet as p
from baseControlEnv import BaseControlEnv

gait_dict = {0: 'Noop', 1: 'Walking', 2: 'Troting', 3:'Pacing', 4:'Pronking', 5:'Bounding', 6:'Static'}

class SoloGaitEnv(BaseControlEnv):
    def __init__(self, config):

        config['rl_dt'] = 0.4
        super(SoloGaitEnv, self).__init__(config)
        
        self.num_actions = 9 # 6 with static
        self.action_space = gym.spaces.Discrete(self.num_actions) # No noop action

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 4 rel foot pose, 6 vel_ref, 10 gait seq = 62
        high = np.inf * np.ones([62])       
        self.observation_space = gym.spaces.Box(-high, high)

    def set_new_gait(self, gait_num):
        # + 1 because 0 corresponds to Noop
        #print('Timestep {}, Gait: {}'.format(self.timestep, gait_dict[gait_num + 1]))
        if self.controller.planner.cg != gait_num + 1:
            self.controller.planner.gait_change = True
            self.controller.planner.cg = gait_num + 1

    def get_observation(self):

        self.robot.UpdateMeasurment()
        qu = np.array([self.robot.baseState[0],
            p.getEulerFromQuaternion(self.robot.baseOrientation)]).flatten()[2:]

        qu_dot = np.array(self.get_base_vel()).flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes

        pfeet = self.get_feet_positions().flatten()

        executed_gait = self.get_past_gait()[:2].flatten()

        history = ...

        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, executed_gait, self.vel_ref.flatten()])
