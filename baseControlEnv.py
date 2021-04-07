import os
import gym
import random
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
Vmax = .7
# External Forces
magnitudes = [3, 5, 8, 10]
durations = [1000, 2000, 3000, 4000, 5000]
def new_random_vel(max_val=Vmax):
    mask = np.array([[1,0,0,0,0,1]])
    vel = (np.random.random((6,1)) - 0.5 ) * 2 * max_val
    vel *= mask.T
    return vel

class BaseControlEnv(gym.core.Env):
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
        self.use_flat_ground = config.get('flat_ground', True)
        self.add_external_force = config.get('add_external_force', False)


        self.velID = 1

        self.rl_dt = config.get('rl_dt', self.T_gait)
        self.k_rl = int(self.rl_dt/self.dt)

        self.N_SIMULATION = int(0.64//self.dt * self.episode_length)

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
                       N_SIMULATION=self.N_SIMULATION,
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

        self.num_actions = None
        self.action_space = None
        self.observation_space = None

        self.continuous_time = 0.0 
        self.discrete_time = 0.0        
        self.auto_vel_switch = config.get('auto_vel_switch', True) 
        self.vel_switch = config.get('vel_switch', 30)#* self.k_rl
        self.use_curriculum = config.get('use_curriculum', False)
        self.max_velocity = 0.0 if self.use_curriculum else Vmax

        self._reset = True
        self._hard_reset = False
        self.timestep = 0
        self._reward_sum = 0
        self._rewards_info = {}
        self._info = {}
        self._info['episode_length'] = 0
        self._info['episode_reward'] = 0
        self._last_action = None
        self.past_commands = deque([np.zeros(3)]*4,maxlen=4)

        if config.get('use_logging', False):
            from soloRL.logger import Logger
            self.logger = Logger(self.N_SIMULATION)
            self.vel_list = np.load('/home/soloRL/misc/vel_plan3.npy')
            self.vel_itr = 0
        else:
            self.logger = None
            self.vel_list = None

    def step(self, action):
        assert not self._reset, "env.reset() must be called before step"
        #assert action < self.num_actions

        self._last_action = action

        self.continuous_time += self.dt
        self.discrete_time += 1
        self.timestep += 1
        self.set_new_gait(np.int(action))
        self.robot.UpdateMeasurment()

        done, info = self.get_termination()
        torque_pen = 0; vel_pen = 0; joints_power = np.zeros(12)
        for _ in range(self.k_rl):
            self._apply_force(self.controller.k) # If noise is added
            self.controller_step()
            self.continuous_time += self.dt
            self.discrete_time += 1
            done, info = self.get_termination()
            # To calculate reward
            torque_pen += np.square(self.robot.tau_ff).sum()
            base_vel = self.get_base_vel().flatten()
            vel_pen += np.square(self.vel_ref.flatten() - base_vel).sum()
            joints_power += self.get_joints_power()
            if self.logger is not None:
                self.log_stats()
            if done:
                break;


        self.switch_velocities()
        self.past_commands.append(self.vel_ref[[0,1,-1]])

        state = self.get_observation()
        #reward = 1 - (1./self.k_rl) * (0.01*torque_pen + vel_pen)
        energy_pen = joints_power.sum() * self.dt 
        reward = 1 - (1./self.k_rl) * (5* energy_pen +  vel_pen)
        if info['nan'] or np.isnan(np.sum(state)):
            state = np.zeros(self.observation_space.shape)
            reward = 0.0
            self._hard_reset = True
            done  = True


        if not self.auto_vel_switch:
            self.vel_ref = self.controller.joystick.v_ref

        self._info['episode_length'] += 1
        self._info['episode_reward'] += reward
        self._info['max_velocity'] = self.max_velocity
        self._info = {**self._info, **info}
        self._info['success'] = info['timeout'] and done

        self._info['dr/Torque_pen'] += torque_pen/self.k_rl
        self._info['dr/body_velocity'] += vel_pen/self.k_rl
        self._info['dr/Energy_pen'] += energy_pen/self.k_rl

        return state, reward, done, self._info.copy()

    def reset(self):
        if self._hard_reset:
            self.reset_hard()
            self._hard_reset = False
        else:
            self.controller.reset()
            self.robot.reset()
        if self.auto_vel_switch: 
            if self.vel_list is not None:
                vel = self.vel_list[0]
                self.vel_itr = 1
            else:
                vel = new_random_vel(self.max_velocity)
            self.reset_vel_ref(vel)
        else:
            self.vel_ref = self.controller.joystick.v_ref

        self.past_commands = deque([np.zeros(3)]*4,maxlen=4)
        self.past_commands.append(self.vel_ref[[0,1,-1]])
        self.create_force_function()

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
        self._info['dr/Energy_pen'] = 0
        self._info['max_velocity'] = self.max_velocity
        self._last_action = None

        if self.logger is not None:
            self.logger.reset()
        
        return self.get_observation()

    def reset_vel_ref(self, vel):
        self.vel_ref = vel
        self.controller.v_ref = vel.reshape(-1,1)

    def close(self):
        self.robot.Stop()
        super().close()

    def get_reward(self, state=None, action=None):
        # reward is height + fwd vel
        torque_pen = 0.5 * np.square(self.robot.tau_ff).sum()

        base_vel = self.get_base_vel().flatten()
        vel_pen = 0.5 * np.square(self.vel_ref.flatten() - base_vel).sum()

        reward = - 0.01 * torque_pen  - 1.0 * vel_pen
        reward = 1 + reward.clip(-10.,0.)

        return reward

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

    def set_new_gait(self, gait_num):
        raise NotImplementedError


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

    def increment_curriculum(self, val=0.1):
        if not self.use_curriculum:
            return
        self.max_velocity = np.clip(self.max_velocity + val, 0.0, Vmax)

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

    def get_observation(self):
        #Basic Observation
        self.robot.UpdateMeasurment()
        
        base_xyz = self.robot.baseState[0]
        base_rpy = p.getEulerFromQuaternion(self.robot.baseOrientation)
        qu = np.array([base_xyz, base_rpy]).flatten()[2:]

        qu_dot = self.get_base_vel().flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes

        pfeet = self.get_feet_positions().flatten()

        executed_gait = self.get_past_gait()[:2].flatten()

        history = ...
        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, executed_gait, self.vel_ref.flatten()])


    def get_termination(self):
        info = {'timeout':False, 'nan': False}
        
        # check for nans
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

    def get_feet_positions(self):
        feet_pos = np.zeros((4,3))
        for i, idx in enumerate(self.feet_ids):
            feet_pos[i] = self.robot_data.oMf[idx].translation
        return feet_pos
            
    def get_current_gait(self):
        return self.controller.planner.Cplanner.get_gait()

    def get_future_gait(self):
        return self.controller.planner.Cplanner.get_gait_des()

    def get_past_gait(self):
        return self.controller.planner.Cplanner.get_gait_past()

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

    @property
    def gait_f(self):
        return self.controller.planner.Cplanner.get_gait()

    @property
    def gait_f_des(self):
        return self.controller.planner.Cplanner.get_gait_des()

    @property
    def gait_p(self):
        return self.controller.planner.Cplanner.get_gait_past()

    def test_validity(self):
        self.reset()
        rs = []
        for i in range(100):
            a = random.randint(0,self.num_actions-1)
            o,r,d,i = self.step(a)
            rs.append(r)
            if d:
                print(sum(rs)/len(rs))
                rs = []
                self.reset()



