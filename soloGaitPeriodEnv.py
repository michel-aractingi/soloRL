import os
import gym
import random
import numpy as np
import pybullet as p
from baseControlEnv import BaseControlEnv
from collections import deque

periods7 =  [0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64]
periods5 =  [0.24, 0.32, 0.4, 0.48, 0.56]
periods4 =  [0.24, 0.32, 0.4, 0.48]

class SoloGaitPeriodEnv(BaseControlEnv):
    def __init__(self, config):

        config['rl_dt'] = 0.32
        super(SoloGaitPeriodEnv, self).__init__(config)
        
        self.num_actions = config.get('num_actions', 4)
        self.action_space = gym.spaces.Discrete(self.num_actions) # No noop action

        self.semi_mdp = config.get('semi_mdp', False)
        self.reactive_update = config.get('reactive_update', False)

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 12 rel foot pose, 6 vel_ref, 4 past gait seq = 62
        high = np.inf * np.ones([62])       
        self.observation_space = gym.spaces.Box(-high, high)

        self.next_period = self.T_gait
        self.past_actions = deque(np.ones(4)*self.next_period,maxlen=4)

        if self.num_actions == 4:
            ps = periods4
        elif self.num_actions == 5:
            ps = periods5
        elif self.num_actions == 7:
            ps = periods7
        else:
            raise NotImplementedError('Invalid number of actions')
        self.period_dict = dict([(i,p) for i,p in enumerate(ps)])

    def reset(self):
        self.next_period = self.T_gait
        self.past_actions = deque(np.ones(4)*self.next_period,maxlen=4)
        self.k_rl = int(self.rl_dt/self.dt)
        return super().reset()

    def set_new_gait(self, action):
        #print(self.period_dict[action])
        period = self.period_dict[action] # No  Noop Actions
        if period != self.next_period:
            self.next_period = period
            if self.reactive_update: # update gait_f
                g,gf = self._update_gait_matrices()
                self.controller.planner.Cplanner.set_gaits(g,gf)
                if self.semi_mdp:
                    self.k_rl = int(period/self.dt)
            else: # update future_gait_des
                self.controller.planner.Cplanner.create_modtrot(period)

        self._last_action = self.next_period
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

        #executed_gaits = self.get_past_gait()[:2].flatten()
        
        command_history = np.stack(self.past_commands).flatten()

        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, history_periods, command_history])

    def _update_gait_matrices(self):
        period = self.next_period
        gait_f = self.get_current_gait()
        gait_p = self.get_past_gait()
    
        # half period MPC steps
        period_steps = int(0.5 * (period /self.dt)/self.k_mpc)
        default_steps = int(0.5 * (self.T_gait/self.dt) /self.k_mpc)
        gait_steps = 2* default_steps

        new_gait_f = np.zeros(gait_f.shape)
        new_gait_f_des = np.zeros(gait_f.shape)

        # If current sequence is still in progess,
        # Start changing time from next sequence
        # Else start changing time of current sequence
        if np.array_equal(gait_f[0,1:], gait_p[0,1:]):
            i_row = 1
            remaining_steps = gait_steps - gait_f[0,0]
            new_gait_f[0,:] = gait_f[0,:]
        else:
            i_row = 0
            remaining_steps = gait_steps

        s1 = gait_f[i_row,1:]
        s2 = 1. - s1
        #s2 = gait_f[i_row + 1,1:]

        seqs = np.vstack((s1,s2))
        i_seq = 0
        remaining_f_steps = 0

        while True:
            if period_steps < remaining_steps:
                new_gait_f[i_row, 0] = period_steps
                new_gait_f[i_row, 1:] = seqs[i_seq]
                remaining_steps -= period_steps
            elif period_steps > remaining_steps:
                new_gait_f[i_row, 0] = remaining_steps
                new_gait_f[i_row, 1:] = seqs[i_seq]
                remaining_f_steps = period_steps - remaining_steps 
                break;
            else:
                new_gait_f[i_row, 0] = period_steps
                new_gait_f[i_row, 1:] = seqs[i_seq]
                remaining_f_steps = 0
                break;
            i_row += 1
            i_seq = (i_seq + 1) % len(seqs)

        # Fill gait_f_des
        if remaining_f_steps!=0:
            new_gait_f_des[0,0] = remaining_f_steps
            new_gait_f_des[0,1:] = seqs[i_seq]
            
            new_gait_f_des[1,0] = period_steps
            new_gait_f_des[1,1:] = seqs[(i_seq + 1) % len(seqs)]

            new_gait_f_des[2,0] = period_steps - remaining_f_steps
            new_gait_f_des[2,1:] = seqs[i_seq]
        else:
            new_gait_f_des[0:2,0] = [period_steps]*2
            last_gait_f_row = new_gait_f[i_row, 1:]
            new_gait_f_des[0,1:] = 1. - last_gait_f_row
            new_gait_f_des[1,1:] = last_gait_f_row

        return new_gait_f, new_gait_f_des
