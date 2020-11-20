import gym
import numpy as np
from soloRL.solo import SoloBase


class SoloBaseEnv(gym.core.Env):
    def __init__(self, config):
        self.robot = SoloBase(config['model_urdf'], 
                              config.get('frame_skip', 4), 
                              control=config.get('control', 'torque'), 
                              task=config.get('task', 'stand'),
                              flat_ground=config.get('flat_ground', True),
                              use_treadmill=config.get('use_treadmill', False),
                              gains=config.get('gains',None),
                              num_history_stack=config.get('num_history_stack', 0),
                              gui=(config['mode']=='gui'))

        self.robot.load()
    
        self.control = config.get('control', 'torque')
        if self.control in ('vpd', 'variable_pd'):
            high = np.ones([len(self.robot.joints) + 2])
        else:
            high = np.ones([len(self.robot.joints)])
        self.action_space = gym.spaces.Box(-high, high)

        high = np.inf * np.ones([self.robot.calc_state().shape[0]])
        self.observation_space = gym.spaces.Box(-high, high)

        self.config = config
        self._reset = True
        self.timestep = 0
        self._reward_sum = 0
        self._rewards_info = {'dr/stand_rew':0,
                              'dr/joint_pose_rew': 0,
                              'dr/torque_rew': 0,
                              'dr/roll_pitch_balance_rew': 0,
                              'dr/progress_rew': 0}

    def step(self, action):
        assert not self._reset, "env.reset() must be called before step"
        
        self.robot.apply_action(action)
        self.simulator_step()
        self.timestep += 1

        state = self.get_observation()
        reward = self.get_reward(state, action)
        done, info = self.is_episode_finished()
        if done:
            self._reset = True
            state = None
            if info['success']:
                if self.robot.task == 'pointgoal':
                    reward = 0.1 * (self.config['episode_length'] - self.timestep)
            else:
                if not info['timeout']: # and self.robot.task !='pointgoal':
                    reward = -10

        self._reward_sum += reward
        info = {**info, **self._rewards_info,
                **{'episode_length':self.timestep,
                   'episode_reward':reward,
                   'goals_reached':self._goals_reached}}

        return state, reward, done, info

    def reset(self):
        self.robot.reset()
        self._reset = False
        self.timestep = 0
        self._reward_sum = 0
        self._goals_reached = 0
        for k in self._rewards_info.keys():
            self._rewards_info[k] = 0.0

        for _ in range(np.random.randint(low=5,high=12)): #prev 20
            self.simulator_step()
        
        return self.get_observation()

    def close(self):
        self.robot.disconnect()
        super().close()

    def get_observation(self):
        return self.robot.calc_state()

    def get_reward(self, state, action=None):
        # reward is height + fwd vel
        if self.robot.task == 'stand':
            # standing reward
            pos_z = self.robot.get_position()[-1]
            stand = float(pos_z > 0.2) * 0.5 # best
            
    
            # Penalty on pose deviation
            poses = np.array([j.get_position() for j in self.robot.ordered_joints])
            jp = -.1 * np.mean(np.abs(poses))

            balance = progress = 0.0
    
        # forward progress reward 
        if self.robot.task == 'walk':
            # standing reward
            pos_z = self.robot.get_position()[-1]
            stand = float(pos_z > 0.2) * 0.5 # best
    
            # Penalty on pose deviation
            poses = np.array([j.get_position() for j in self.robot.ordered_joints])
            jp = -.1 * np.mean(np.square(poses))
            
            if pos_z > 0.2:
                vx = self.robot.get_linear_velocity()[0] 
                progress =  2 * np.sign(vx) * vx**2
            else:
                progress = 0.0

            balance = 0.0

        if self.robot.task == 'pointgoal':
            # standing reward
            pos_z = self.robot.get_position()[-1]
            stand = float(pos_z > 0.2) * 0.5 # best
    
            # Penalty on pose deviation
            poses = np.array([j.get_position() for j in self.robot.ordered_joints])
            jp = -.1 * np.mean(np.square(poses))

            # roll pitch deviation penalty
            roll, pitch, _ = self.robot.get_orientation()
            balance = -0.1 * (np.abs(roll) + np.abs(pitch))

            if pos_z > 0.2:
                progress = 1 * self.robot.progress * (1.0 / self.robot.scene.dt)
            else:
                progress = 0.0

        # torque penalty
        if self.config.get('control', 'torque') == 'torque':
            tp = np.square(action).sum()
            torque  = -0.01 * tp

        reward = stand + jp + balance + progress + torque
        self.update_rewards_info(stand, jp, balance, progress, torque)

        # feet off ground penalty
        #fg_contact = self.robot.get_feet_ground_contact()
        #r += -0.05 * (1.0 - fg_contact).sum() # no effect on stand

        # trajectory tracking
        #if pd in self.config['control']:
             #r += -0.1 * np.linalg.norm(poses - self.old_pose_des)

        return reward

    def simulator_step(self):
        self.robot.simulator_step()

    def is_episode_finished(self):
        info = {}
        if self.timestep >= self.config['episode_length']:
            info['timeout'] = True
            info['success'] = self.robot.task!='pointgoal'
            return True, info

        if self.robot.get_position()[-1] < 0.05:
            info['timeout'] = False
            info['success'] = False
            return True, info

        if self.robot.task=='pointgoal' and self.robot.goals_reached > self._goals_reached: # goal reached
            self._goals_reached = self.robot.goals_reached
            info['success'] = True
            info['timeout'] = False
            return True, info

        return False, info

    def update_rewards_info(self, stand, jp, balance, progress, torque):
        self._rewards_info['dr/stand_rew'] += stand
        self._rewards_info['dr/joint_pose_rew'] += jp
        self._rewards_info['dr/roll_pitch_balance_rew'] += balance
        self._rewards_info['dr/torque_rew'] += torque
        self._rewards_info['dr/progress_rew'] += progress
