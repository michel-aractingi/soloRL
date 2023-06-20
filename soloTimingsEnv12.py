import os
import gym
import random
import yaml
import numpy as np
import pybullet as p
from collections import deque

from scripts import Controller, PyBulletSimulator

"""
Constants
"""
coulomb_tau = 0.0477
viscous_b = 0.000135
K_motor = 4.81
feet_frames_name = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
"""
-------------------------------------------------------------
"""
MAXFORCE=10
mask = np.array([[1,0,0,0,0,0]])
Vmax = 0.3
def new_random_vel(max_val=Vmax):
    #vel = (np.random.random((6,1)) - 0.5 ) * 2 * max_val
    vel = np.random.random((6,1)) * max_val
    vel *= mask.T
    return vel

max_timing = 0.26

def get_oscillator_function(tnc, tc, freq):
    assert freq >= tc and tc > tnc
    tc = np.int(tc); tnc = np.int(tnc)
    v = np.ones((freq,), dtype=np.float32)
    v[tnc:tc] = 0.0
    return np.concatenate([v]*500)

class SoloTimingsEnv12:
    def __init__(self, config):

        el = config['episode_length']
        config = config['robot']
        config['episode_length'] = el

        config['rl_dt'] = config.get('dt_mpc', 0.02)

        self.mode = 'gui' if config.get('enable_pyb_GUI', False) else 'headless'
        self.num_history_stack = config.get('num_history_stack', 1)
        self.add_external_force = config.get('add_external_force', False)
        self.use_flat_ground = config.get('use_flat_plane', True)
        self.q_init = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])

        self.controller = \
            Controller(q_init=self.q_init, 
                       envID=0,
                       velID=1,
                       dt_wbc=config['dt_wbc'],
                       dt_mpc=config['dt_mpc'], 
                       k_mpc=int(config['dt_mpc'] / config['dt_wbc']),
                       t=0,
                       T_gait=config['T_gait'],
                       T_mpc=config['T_mpc'],
                       N_SIMULATION=config['N_SIMULATION'],
                       type_MPC=config['type_MPC'],
                       use_flat_plane= self.use_flat_ground,
                       predefined_vel=True,
                       kf_enabled=config['kf_enabled'],
                       N_gait=config['N_gait'],
                       enable_pyb_GUI=self.mode=='gui',
                       isSimulation=True)

        self.robot = PyBulletSimulator()
        self.robot.Init(calibrateEncoders=True, 
                        q_init=self.q_init,
                        envID=0,
                        use_flat_plane=self.use_flat_ground,
                        enable_pyb_GUI=self.mode=='gui',
                        dt=config['dt_wbc'])

        self.robot_model = self.controller.myController.invKin.rmodel
        self.robot_data = self.controller.myController.invKin.rdata
        self.feet_ids = [self.robot_model.getFrameId(n) for n in feet_frames_name]

        self.num_actions = 12 # 8 describing new gait
        self.action_space = gym.spaces.Box(shape=(self.num_actions,), high=1, low=0)
        #(tnc, duty cycle d)**i#

        # 1 base pose z, 3 base orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 12 rel foot pose, 3 vel_ref, 4 current contacts and 4 next contacts = 57
        self.flat_observation = config.get('flat_observation', False)
        if self.flat_observation:
            self._obs_size = 69
            high = np.inf * np.ones((self.num_history_stack * self._obs_size))
        else:
            self._obs_size = 72
            high = np.inf * np.ones((self.num_history_stack, self._obs_size))       
        self.observation_space = gym.spaces.Box(-high, high)

        self.action_history = deque([np.zeros((self.num_actions,))]*self.num_history_stack,
                                    maxlen=self.num_history_stack)
        self.observation_history = deque([np.zeros((self._obs_size,))]*self.num_history_stack,                                         maxlen=self.num_history_stack)

        self.dt = config.get('dt', 0.002)

        self.episode_length = config.get('episode_length', 1000)
        self.auto_vel_switch = config.get('auto_vel_switch', True)
        self.vel_switch = config.get('vel_switch', 1000)#* self.k_rl
        self.use_curriculum = config.get('use_curriculum', False)
        self.max_velocity = 0.0 if self.use_curriculum else Vmax

        self._hard_reset = False
        self._reset = True
        self.timestep = 0
        self._info = {}
        self._info['episode_length'] = 0
        self._info['episode_reward'] = 0
        self._last_action = None

        self.config = config

        if self.add_external_force:
            if self.use_curriculum:
                self.min_max_force = np.array([0,2])
            else:
                self.min_max_force = DEFAULTFORCE
        else:
            self.min_max_force = np.zeros(2)
        
        if config.get('use_logging', False):
            from soloRL.logger import Logger
            self.logger = Logger(self.N_SIMULATION)
            self.vel_list = None#np.load('/home/soloRL/misc/vel_plan4.npy')
            self.vel_itr = 0
        else:
            self.logger = None
            self.vel_list = None

    def reset(self):
        if self._hard_reset:
            self.reset_hard()
            self._hard_reset = False

        self.controller.reset()
        self.robot.reset()

        self.action_history = deque([np.zeros((self.num_actions,))]*self.num_history_stack,
                                    maxlen=self.num_history_stack)
        self.observation_history = deque([np.zeros((self._obs_size,))]*self.num_history_stack,                                         maxlen=self.num_history_stack)

        vel = new_random_vel(self.max_velocity)
        self.reset_vel_ref(vel)
        #if self.auto_vel_switch: 
        #    if self.vel_list is not None:
        #        vel = self.vel_list[0]
        #        self.vel_itr = 1
        #    else:
        #else:
        #    self.vel_ref = self.controller.joystick.v_ref

        if self.logger is not None:
            self.logger.reset()
        
        self._last_action = np.array([8,8,0, 0,8,8, 0,8,8, 8,8,0], dtype=np.int)
        self._contacts = [get_oscillator_function(8,16,16),
                          get_oscillator_function(0,8,16),
                          get_oscillator_function(0,8,16),
                          get_oscillator_function(8,16,16)]
        self.store_actions((self._last_action * self.dt_mpc /  max_timing) )

        self.store_observation()


        self.create_force_function()
        self.timestep = 0
        self.continuous_time = 0

        self._info['episode_length'] = 0
        self._info['episode_reward'] = 0
        self._info['dr/Torque_pen'] = 0
        self._info['dr/body_velocity'] = 0
        self._info['dr/Energy_pen'] = 0

        self._info['max_velocity'] = self.max_velocity
        self._info['max_force'] = self.min_max_force[1]
        self._info['min_force'] = self.min_max_force[0]
        #self._last_action = None
        self._reset = False

        return self.get_observation()

    def step(self, action):
        assert len(action) == self.num_actions and not self._reset
        assert self.controller.k % self.k_mpc == 0

        action = (action.clip(-1,1) + 1) * 0.5
        contact_config = ((action * max_timing)/self.dt_mpc).astype(np.int)

        for l in range(0,len(contact_config), 3):
            if not np.array_equal(contact_config[l:l+3], self._last_action[l:l+3]):
                tnc, d, f = contact_config[l:l+3].flatten()
                tc = tnc + d if d!=0 else tnc + 1
                if tnc == f == 0: f = 1
                freq= np.int(np.clip(tc + f, 4, 2*max_timing/self.dt_mpc))
                self._contacts[int(l/3)] = get_oscillator_function(tnc, tc, freq)
            
        # construct gait and check for errors cases (TODO hacky)
        gait = np.vstack([c[:self.N_gait] for c in self._contacts]).T
        if np.sum(gait) == 0.0:
            self._reset = True
            return None, -1, True, self._info

        self.set_new_gait(gait)

        torque_pen = vel_pen = joints_power = 0.0;
        for i in range(self.k_mpc):
            self._apply_force(self.controller.k) # If noise is added     
            self.controller_step()

            self.continuous_time += self.dt
            done, info = self.get_termination()

            # Collect reward measures
            torque_pen += np.square(self.robot.tau_ff).sum()
            base_vel = self.get_base_vel().flatten()
            vel_pen += np.square(self.vel_ref.flatten() - base_vel).sum()
            joints_power += self.get_joints_power()

        self.store_actions(action)
        self.store_observation()
        self._last_action = contact_config

        self.timestep += 1

        # Compute reward
        energy_pen = joints_power.sum() * self.dt 
        reward = 1. - (10* energy_pen +  vel_pen)/self.k_mpc

        if self.auto_vel_switch and self.vel_switch % self.timestep == 0:
            self.switch_velocities()

        done, info = self.get_termination()

        observation = self.get_observation()
        if info['nan'] or np.isnan(np.sum(observation)) or np.isnan(reward):
            #print('obs: ',self.get_observation(), 'rew :', reward, 'inf: ', self._info.copy())
            #print('act :', action)
            observation = None
            reward = -10
            done = True

        self._info['episode_length'] += 1
        self._info['episode_reward'] += reward
        self._info = {**self._info, **info}
        self._info['success'] = info['timeout'] and done
        self._info['max_velocity'] = self.max_velocity

        self._info['dr/Torque_pen'] += torque_pen
        self._info['dr/body_velocity'] += vel_pen
        self._info['dr/Energy_pen'] += energy_pen

        #obs_action_dict = {'obs':self.get_observation(), 'action':self.get_action_history()}

        self._reset = done 
        return observation, reward, done, self._info.copy()

    def controller_step(self):

        self.robot.UpdateMeasurment()

        # Desired torques
        self.controller.compute(self.robot)

        # Set desired quantities for the actuators
        self.robot.SetDesiredJointPDgains(self.controller.result.P, self.controller.result.D)
        self.robot.SetDesiredJointPosition(self.controller.result.q_des)
        self.robot.SetDesiredJointVelocity(self.controller.result.v_des)
        self.robot.SetDesiredJointTorque(self.controller.result.tau_ff.ravel())

        # Step simulation for one dt
        self.robot.SendCommand(WaitEndOfCycle=False)

    def set_new_gait(self, gait):
        self.controller.gait.setGait(gait)
        self.advance_contacts()

    def advance_contacts(self):
        for i in range(4):
            self._contacts[i] = np.roll(self._contacts[i],-1)

    def get_observation(self):
        """
        return an observation of size Nhistory x Nfeatures
        """
        obs = np.stack(self.observation_history).copy()
        if self.flat_observation:
            return obs.flatten()
        else:
            return obs

    def store_observation(self):
        self.robot.UpdateMeasurment()
        internal_state = self.get_internal_state()
        current_contacts = self.get_current_gait()[:2,:].flatten() # or past TODO

        vel_ref = self.vel_ref.flatten()[[0,1,-1]]
 
        observation = np.concatenate((internal_state, vel_ref, current_contacts, self._last_action))

        if not self.flat_observation:
            observation = np.concatenate((observation, np.zeros((3,)))) 
            
        self.observation_history.append(observation)

    def reset_vel_ref(self, vel):
        #vel[0,0]=0.3
        self.vel_ref = vel
        self.controller.v_ref = vel.reshape(-1,1)

    def close(self):
        self.robot.Stop()
        #super().close()

    def store_actions(self, actions):
        self.action_history.append(actions)

    def get_action_history(self):
        return np.stack(self.action_history).copy()

    @property
    def N_gait(self):
        return self.config['N_gait']

    @property
    def dt_mpc(self):
        return self.config['dt_mpc']

    @property
    def k_mpc(self):
        return self.controller.k_mpc

    ######################################################


    def get_joints_power(self):
        """
        P = P_t + P_f
        P_f: power loss due to friction, tau_f * qa_dot
        P_t: power due to torques, K * tau**2
        ------------------------------------------
        tau_f: friction torque, tau_c * sign(qa_dot) + b * qa_dot
        tau_c: columb friction 
        b: viscous friction
        K: scale motor resistance
        These constant values are provided by the lab
        """
        qa_dot = self.robot.v_mes
        tau_cmd = self.robot.tau_ff

        tau_friction = coulomb_tau * np.sign(qa_dot) + viscous_b * qa_dot

        P_f = tau_friction * qa_dot 
        P_t = K_motor * tau_cmd**2

        return P_f + P_t

    def get_base_vel(self):
        '''
        return the base linear and angular velocities in the body frame
        '''
        return np.concatenate((self.robot.b_baseVel, self.robot.baseAngularVelocity)).reshape((-1,1))

    def get_feet_positions(self):
        feet_pos = np.zeros((4,3))
        for i, idx in enumerate(self.feet_ids):
            feet_pos[i] = self.robot_data.oMf[idx].translation
        return feet_pos

    def get_internal_state(self):
        qu = np.array([self.robot.baseState[0],
            p.getEulerFromQuaternion(self.robot.baseOrientation)]).flatten()[2:]
        qu_dot = np.array(self.get_base_vel()).flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes
        pfeet = self.get_feet_positions().flatten()

        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet])

    def get_termination(self):
        info = {'timeout':False, 'nan': False}

        if self.controller.error_flag == 4:
            print('nan detected')
            info['nan'] = True
            self._hard_reset = True
            return True, info

        # if fallen
        if self.robot.baseState[0][-1] < 0.11 or self.controller.myController.error:
            return True, info

        if self.timestep >= self.episode_length:
            info['timeout'] = True
            return True, info

        return False, info

    def create_force_function(self):
        if not self.add_external_force:
            self._apply_force = lambda k: None
        else:
            M = np.zeros((3,))
            F = np.zeros((3,))
            F[random.choices([0,1,2])] = np.random.randint(self.min_max_force[0],
                                                           self.min_max_force[1] + 1)
            sign = random.choices([-1,1])[0]
            F *= np.array([sign, sign, 1.])
            start_itr = random.randint(500, int(self.k_rl * self.episode_length *(2/3)))
            duration = random.choice(durations)
            #print('apply force with magniture {} starting at iteration {} for a duration of {} steps'.format(F,start_itr, duration))
            self._apply_force = lambda k: self.robot.pyb_sim.apply_external_force(k, start_itr, duration, F, M)

    def log_stats(self):
        if self.logger is None:
            return
        #Basic Observation
        self.robot.UpdateMeasurment()

        base_xyz = self.robot.baseState[0]
        base_rpy = p.getEulerFromQuaternion(self.robot.baseOrientation)

        joints_power = self.get_joints_power()

        self.logger.log(self.controller.k,
                        self.get_base_vel().flatten(),
                        self.vel_ref.flatten(),
                        self.robot.tau_ff,
                        joints_power,
                        base_xyz, base_rpy, self._last_action)

    def switch_velocities(self):
        if self.auto_vel_switch and self.timestep % self.vel_switch == 0:
            if self.vel_list is not None:
                vel = self.vel_list[self.vel_itr]
                self.vel_itr = (self.vel_itr + 1) % self.vel_list.shape[0]
            else:
                vel = new_random_vel(self.max_velocity)
            self.reset_vel_ref(vel)
        else:
            return

    def increment_curriculum(self, val=0.05):
        if not self.use_curriculum:
            return
        self.max_velocity = np.clip(self.max_velocity + val, 0.0, Vmax)

        #Increment min and max force
        self.min_max_force  = np.clip(self.min_max_force + 1,
                                      np.array([0,0]),
                                      np.array([MAXFORCE-2, MAXFORCE]))


    def get_current_gait(self):
        return self.controller.gait.getCurrentGait()


    def reset_hard(self):
        self.controller = \
            Controller(q_init=self.q_init, 
                       envID=0,
                       velID=1,
                       dt_wbc=self.config['dt_wbc'],
                       dt_mpc=self.config['dt_mpc'], 
                       k_mpc=int(self.config['dt_mpc'] / self.config['dt_wbc']),
                       t=0,
                       T_gait=self.config['T_gait'],
                       T_mpc=self.config['T_mpc'],
                       N_SIMULATION=self.config['N_SIMULATION'],
                       type_MPC=self.config['type_MPC'],
                       use_flat_plane= self.use_flat_ground,
                       predefined_vel=True,
                       kf_enabled=self.config['kf_enabled'],
                       N_gait=self.config['N_gait'],
                       enable_pyb_GUI=self.mode=='gui',
                       isSimulation=True)

        self.robot = PyBulletSimulator()
        self.robot.Init(calibrateEncoders=True, 
                        q_init=self.q_init,
                        envID=0,
                        use_flat_plane=self.use_flat_ground,
                        enable_pyb_GUI=self.mode=='gui',
                        dt=self.config['dt_wbc'])
