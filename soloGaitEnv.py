import os
import gym
import numpy as np
import pybullet as p

os.chdir('/home/quadruped-reactive-walking-feature-merge-mpc-tsid/scripts/')
from Controller import Controller
from PyBulletSimulator import PyBulletSimulator


feet_frames_name = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']

class SoloGaitEnv(gym.core.Env):
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
        self.episode_length = config.get('episode_length', 50)
        self.use_flat_ground = config.get('flat_ground', True)

        self.rl_dt = 0.4

        self.controller = \
            Controller(q_init=self.q_init, 
                       envID=0,
                       velID=1,
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

        self.num_gaits = 6
        self.reset_vel_ref(np.array([0.5,0,0,0,0,0]))#vel_ref)
        self.action_space = gym.spaces.Discrete(self.num_gaits)

        # 1 base pose z, 3 orn , 6 body vel, 12 Joint angles , 12 Joints Vel,  
        # 4 rel foot pose, 6 vel_ref, 10 gait seq = 62
        high = np.inf * np.ones([62])       
        self.observation_space = gym.spaces.Box(-high, high)

        self.continuous_time = 0.0 
        self.discrete_time = 0.0

        self._reset = True
        self.timestep = 0
        self._reward_sum = 0
        self._rewards_info = {}

    def step(self, action):
        assert not self._reset, "env.reset() must be called before step"

        self.continuous_time += self.dt
        self.discrete_time += 1
        self.timestep += 1
        self.set_new_gait(action)
        self.robot.UpdateMeasurment()

        done, info = self.get_termination()
        while self.discrete_time % (self.rl_dt/self.dt)!=0 and not done:
            print(self.continuous_time % self.rl_dt)
            self.controller_step()
            self.continuous_time += self.dt
            self.discrete_time += 1
            done, info = self.get_termination()

        state = self.get_observation()
        reward = self.get_reward()

        return state, reward, done, {}#self._info

    def reset(self):
        self.robot.reset()
        self._reset = False
        self.timestep = 0
        self.continuous_time = 0
        self._reward_sum = 0
        self._goals_reached = 0
        for k in self._rewards_info.keys():
            self._rewards_info[k] = 0.0

        for _ in range(np.random.randint(low=5,high=12)): #prev 20
            self.simulator_step()
        
        return self.get_observation()

    def reset_vel_ref(self, vel):
        self.vel_ref = vel
        self.controller.v_ref = vel.reshape(-1,1)

    def close(self):
        #self.robot.disconnect()
        p.disconnect()
        super().close()

    def get_reward(self, state=None, action=None):
        # reward is height + fwd vel

        torque_pen = 0.5 * np.square(self.robot.tau_ff).sum()

        base_vel = np.array(self.robot.baseVel).flatten()
        vel_pen = 0.5 * np.square(self.vel_ref - base_vel).sum()

        reward = 1  - 0.01 * torque_pen  - 0.5 * vel_pen

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
        self.controller.planner.cg = gait_num
        self.controller.planner.gait_change = True

    def get_observation(self):

        self.robot.UpdateMeasurment()
        qu = np.array([self.robot.baseState[0],
            p.getEulerFromQuaternion(self.robot.baseOrientation)]).flatten()[2:]

        qu_dot = np.array(self.robot.baseVel).flatten()
        qa = self.robot.q_mes
        qa_dot = self.robot.v_mes

        pfeet = self.get_feet_positions().flatten()

        executed_gait = self.get_past_gait()[:2].flatten()

        history = ...

        return np.concatenate([qu, qu_dot, qa, qa_dot, pfeet, executed_gait, self.vel_ref])


    def get_termination(self):
        info = {}
        # if fallen
        if self.robot.baseState[0][-1] < 0.05:
            info['timeout'] = False
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
