import os
import gym
import random
import numpy as np
import pybullet as p
from baseControlEnv import BaseControlEnv
from collections import deque

period_dict = {0: -1, 1: 0.24, 2: 0.32, 3: 0.40, 4:0.48}

class SoloGaitPeriodEnv(BaseControlEnv):
    def __init__(self, config):

        config['rl_dt'] = 0.32
        super(SoloGaitPeriodEnv, self).__init__(config)
        
        self.num_actions = 4 
        self.action_space = gym.spaces.Discrete(self.num_actions) # No noop action

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 12 rel foot pose, 6 vel_ref, 4 past gait seq = 62
        high = np.inf * np.ones([66])       
        self.observation_space = gym.spaces.Box(-high, high)

        self.next_period = self.T_gait
        self.past_actions = deque(np.ones(4)*self.next_period,maxlen=4)

    def reset(self):
        self.past_actions = deque(np.ones(4)*-1,maxlen=4)
        self.current_period = self.T_gait
        return super().reset()

    def set_new_gait(self, action):
        period = period_dict[action + 1] # No  Noop Actions
        if period != self.next_period:
            self.controller.planner.Cplanner.create_modtrot(period)
            self.next_period = period
        self.past_actions.append(self.next_period)

    def get_observation(self):

        self.robot.UpdateMeasurment()
        qu = np.array([self.robot.baseState[0],
            p.getEulerFromQuaternion(self.robot.baseOrientation)]).flatten()[2:]

        qu_dot = np.array(self.get_base_vel()).flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes

        pfeet = self.get_feet_positions().flatten()
    
        history_periods = np.array(self.past_actions) 

        executed_gaits = self.get_past_gait()[:2].flatten()

        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, history_periods, executed_gaits, self.vel_ref.flatten()])
