import os
import gym
import random
import numpy as np
import pybullet as p
from baseControlEnv import BaseControlEnv
from collections import deque

feet_frames_name = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']

gait_dict = {0: [1.,1.,1.,1.],
             1: [1.,1.,1.,0.], 
             2: [1.,1.,0.,1.], 
             3: [1.,0.,1.,1.], 
             4: [0.,1.,1.,1.], 
             5: [1.,0.,1.,0.], 
             6: [0.,1.,0.,1.], 
             7: [1.,0.,0.,1.], 
             8: [0.,1.,1.,0.],
             -1: [0.,0.,0.,0.]}

gait_name_dict = {0:'Static', 1:'Walk1', 2:'Walk2', 3:'Walk3', 4:'Walk4', 5:'Pace1', 6:'Pace2', 7:'Trot1', 8:'Trot2'}

class SoloGaitEnvContact(BaseControlEnv):
    def __init__(self, config):

        self.T_gait = config.get('T_gait', 0.32)
        config['rl_dt'] = self.T_gait / 2
        super(SoloGaitEnvContact, self).__init__(config)

        self.past_gaits = deque([-1,-1,-1], maxlen=3) # get past 3 gaits

        self.num_actions = 9
        self.action_space = gym.spaces.Discrete(self.num_actions) # No noop action

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 4 rel foot pose, 6 vel_ref, 10 gait seq = 62
        high = np.inf * np.ones([64])       
        self.observation_space = gym.spaces.Box(-high, high)

    def step(self, action):
        self.past_gaits.append(np.int(action))
        return super().step(action)

    def set_new_gait(self, gait_num):
        #print('Timestep {}, Contact Seq: {} {}'.format(self.timestep, gait_name_dict[gait_num], gait_num))
        self.controller.planner.gait_change = True
        self.controller.planner.cg = gait_num #+ 1 

    def get_observation(self):
        self.robot.UpdateMeasurment()
        qu = np.array([self.robot.baseState[0],
            p.getEulerFromQuaternion(self.robot.baseOrientation)]).flatten()[2:]

        qu_dot = np.array(self.get_base_vel()).flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes

        pfeet = self.get_feet_positions().flatten()

        executed_past_seq = np.array([gait_dict[i] for i in self.past_gaits], dtype=np.float32).flatten()
        history = ...
        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, executed_past_seq, self.vel_ref.flatten()])
