import os
import gym
import random
import numpy as np
import pybullet as p

#os.chdir('/home/quadruped-reactive-walking-feature-merge-mpc-tsid/scripts/')
from scripts import Controller, PyBulletSimulator
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

Vmax = 0.5
gait_name_dict = {0:'Static', 1:'Walk1', 2:'Walk2', 3:'Walk3', 4:'Walk4', 5:'Pace1', 6:'Pace2', 7:'Trot1', 8:'Trot2'}

# External Forces
magnitudes = [0, 3, 5, 10, 12]
durations = [1000, 2000, 3000, 4000, 5000]

def new_random_vel():
    mask = np.array([[1,0,0,0,0,0]])
    vel = (np.random.random((1,6)) - 0.5 ) * 2 * Vmax
    vel *= mask
    return vel


class SoloGaitEnvContact(gym.core.Env):
    def __init__(self, config):
        
        self.config = config
        self.dt = config.get('dt', 0.002)
        self.mode = config.get('mode', 'headless')
        #self.q_init = config.q_init
        self.q_init = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])

        self.dt_wbc = config.get('dt_wbc', self.dt)
        self.dt_mpc = config.get('dt_mpc', 0.02)

        self.k_mpc = int(self.dt_mpc//self.dt_wbc)

        self.T_gait = config.get('T_gait', 0.32)
        self.T_mpc  = config.get('T_mpc', 0.32)

        self.solo12 = config.get('solo12', True)
        self.episode_length = config.get('episode_length', 100)
        self.vel_switch = config.get('vel_switch', 30)
        self.use_flat_ground = config.get('flat_ground', True)
        self.auto_vel_switch = config.get('auto_vel_switch', True)
        self.add_external_force = config.get('add_external_force', False)

        self.velID = 1

        self.rl_dt = self.T_gait / 2#0.32 # .4
        self.k_rl = int(self.rl_dt/self.dt)

        self.controller = \
            Controller(q_init=self.q_init, 
                       envID=0,
                       velID=self.velID,
                       dt_tsid=self.dt_wbc,
                       dt_mpc=self.dt_mpc, 
                       k_mpc=self.k_mpc,
                       t=0,
                       T_gait=self.T_gait,
                       T_mpc=self.T_mpc,
                       N_SIMULATION=50000,
                       type_MPC=True,
                       pyb_feedback=True, 
                       on_solo8= not self.solo12,
                       use_flat_plane= self.use_flat_ground,
                       predefined_vel=True,
                       enable_pyb_GUI=self.mode=='gui')

        self.robot = PyBulletSimulator()
        self.robot.Init(calibrateEncoders=True, 
                        q_init=self.q_init,
                        envID=0,
                        use_flat_plane=self.use_flat_ground,
                        enable_pyb_GUI=self.mode=='gui',
                        dt=self.dt_wbc)

        self.robot_model = self.controller.myController.invKin.rmodel
        self.robot_data = self.controller.myController.invKin.rdata
        self.feet_ids = [self.robot_model.getFrameId(n) for n in feet_frames_name]
        self.past_gaits = deque([-1,-1,-1], maxlen=3) # get past 3 gaits

        self.num_gaits = 9
        self.action_space = gym.spaces.Discrete(self.num_gaits) # No noop action

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 4 rel foot pose, 6 vel_ref, 10 gait seq = 62
        high = np.inf * np.ones([64])       
        self.observation_space = gym.spaces.Box(-high, high)

        self.continuous_time = 0.0 
        self.discrete_time = 0.0

        self._reset = True
        self._hard_reset = False
        self.timestep = 0
        self._reward_sum = 0
        self._rewards_info = {}
        self._info = {}
        self._info['episode_length'] = 0
        self._info['episode_reward'] = 0

    def step(self, action):
        assert not self._reset, "env.reset() must be called before step"
        assert action < self.num_gaits

        self.continuous_time += self.dt
        self.discrete_time += 1
        self.timestep += 1
        self.set_new_gait(np.int(action))
        self.robot.UpdateMeasurment()

        self.past_gaits.append(np.int(action))

        done, info = self.get_termination()
        torque_pen = 0; vel_pen = 0;
        for i in range(self.k_rl):
            self._apply_force(self.controller.k) # If noise is added
            err = self.controller_step()
            self.continuous_time += self.dt
            self.discrete_time += 1
            done, info = self.get_termination()
            # To calculate reward
            torque_pen += np.square(self.robot.tau_ff).sum()
            base_vel = self.get_base_vel().flatten()
            vel_pen += np.square(self.vel_ref.flatten() - base_vel).sum()
            if done:
                break;

        state = self.get_observation()
        reward = 1 - (1./self.k_rl) * (0.005*torque_pen + vel_pen)
        #print('Rewards vel:', vel_pen/self.k_rl, '  Torque: ', torque_pen/self.k_rl)
        #reward = self.get_reward()
        if info['nan'] or np.isnan(np.sum(state)):
            state = np.zeros(self.observation_space.shape)
            reward = 0.0
            self._hard_reset = True
            done  = True

        if not self.auto_vel_switch:
            self.vel_ref = self.controller.joystick.v_ref

        self._info['episode_length'] += 1
        self._info['episode_reward'] += reward
        self._info = {**self._info, **info}
        self._info['success'] = info['timeout'] and done

        self._info['dr/Torque_pen'] += torque_pen/self.k_rl
        self._info['dr/body_velocity'] += vel_pen/self.k_rl

        # Change velocity command every N steps
        if self.auto_vel_switch and self.timestep % self.vel_switch == 0: 
            self.reset_vel_ref(new_random_vel())
        
        #print('gait_f \n',self.controller.planner.Cplanner.get_gait()[:6])
        #print('gait_f_des \n',self.controller.planner.Cplanner.get_gait_des()[:6])

        return state, reward, done, self._info.copy()

    def reset(self):
        if self._hard_reset or self.controller.error_flag ==4:
            print('reset_hard')
            self.reset_hard()
            self._hard_reset = False
        else:
            self.controller.reset()
            self.robot.reset()
        if self.auto_vel_switch: 
            self.reset_vel_ref(new_random_vel())
        else:
            self.vel_ref = self.controller.joystick.v_ref
        #self.reset_vel_ref(np.array([[2,0,0,0,0,0]]).T)

        self._reset = False
        self.timestep = 0
        self.continuous_time = 0
        self.discrete_time = 0
        self._reward_sum = 0
        self._goals_reached = 0
        self._info['episode_length'] = 0
        self._info['episode_reward'] = 0
        self._info['dr/Torque_pen'] = 0
        self._info['dr/body_velocity'] = 0

        self.create_force_function()

        return self.get_observation()

    def reset_vel_ref(self, vel):
        vel = np.array([[0.25,0,0,0,0,0]])
        self.vel_ref = vel
        self.controller.v_ref = vel.T

    def close(self):
        self.robot.Stop()
        super().close()

    def get_reward(self, state=None, action=None):
        # reward is height + fwd vel
        torque_pen = 0.5 * np.square(self.robot.tau_ff).sum()

        base_vel = self.get_base_vel().flatten()
        vel_pen = 0.5 * np.square(self.vel_ref.flatten() - base_vel).sum()

        reward = - 0.01 * torque_pen  - 2.0 * vel_pen
        reward = 1 + reward.clip(-10.,0.)
        #print('Velocity difference', vel_pen, reward)

        return reward

    def controller_step(self):

        self.robot.UpdateMeasurment()

        # Desired torques
        self.controller.compute(self.robot)
        if self.controller.error_flag == 4:
            print('error')
            return 1

        # Set desired quantities for the actuators
        self.robot.SetDesiredJointPDgains(self.controller.result.P, self.controller.result.D)
        self.robot.SetDesiredJointPosition(self.controller.result.q_des)
        self.robot.SetDesiredJointVelocity(self.controller.result.v_des)
        self.robot.SetDesiredJointTorque(self.controller.result.tau_ff.ravel())

        # Step simulation for one dt
        self.robot.SendCommand(WaitEndOfCycle=False)
        return 0

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

    def reset_hard(self):
       
        del self.controller, self.robot
        self.controller = \
            Controller(q_init=self.q_init, 
                       envID=0,
                       velID=self.velID,
                       dt_tsid=self.dt_wbc,
                       dt_mpc=self.dt_mpc, 
                       k_mpc=self.k_mpc,
                       t=0,
                       T_gait=self.T_gait,
                       T_mpc=self.T_mpc,
                       N_SIMULATION=50000,
                       type_MPC=True,
                       pyb_feedback=True, 
                       on_solo8= not self.solo12,
                       use_flat_plane= self.use_flat_ground,
                       predefined_vel=True,
                       enable_pyb_GUI=self.mode=='gui')
        
        #self.controller.reset()
        self.robot = PyBulletSimulator()
        self.robot.Init(calibrateEncoders=True, 
                        q_init=self.q_init,
                        envID=0,
                        use_flat_plane=self.use_flat_ground,
                        enable_pyb_GUI=self.mode=='gui',
                        dt=self.dt_wbc)

    def get_termination(self):
        info = {'timeout':False, 'nan': False}
        # if fallen
        if self.robot.baseState[0][-1] < 0.11 or self.controller.myController.error:
            return True, info

        # check for nans
        if self.controller.error_flag == 4:
            print('nan detected')
            info['nan'] = True
            self._hard_reset = True
            return True, info

        if self.timestep >= self.episode_length:
            info['timeout'] = True
            return True, info

        return False, info

    def get_feet_positions(self):
        feet_pos = np.zeros((4,3))
        for i, idx in enumerate(self.feet_ids):
            feet_pos[i] = self.robot_data.oMf[idx].translation
        return feet_pos
            
    def get_past_gait(self):
        return self.controller.planner.Cplanner.get_gait_past()

    def get_base_vel(self):
        '''
        return the base linear and angular velocities in the body frame
        '''
        return np.concatenate((self.robot.b_baseVel, self.robot.baseAngularVelocity)).reshape((-1,1))

    def create_force_function(self):
        if not self.add_external_force:
            self._apply_force = lambda k: None
        else:
            M = np.zeros((3,))
            F = np.zeros((3,))
            F[random.choices([0,1,2])] = random.choices(magnitudes) 
            sign = random.choices([-1,1])[0]
            F *= np.array([sign, sign, 1.])
            start_itr = random.randint(500, int(self.k_rl * self.episode_length *(2/3)))
            duration = random.choice(durations)
            print('apply force with magniture {} starting at iteration {} for a duration of {} steps'.format(F,start_itr, duration))
            self._apply_force = lambda k: self.robot.pyb_sim.apply_external_force(k, start_itr, duration, F, M)


    def test_validity(self):
        self.reset()
        for i in range(100):
            a = random.randint(0,self.num_gaits-1)
            o,r,d,i = self.step(a)
            if d:
                print(r)
                self.reset()



