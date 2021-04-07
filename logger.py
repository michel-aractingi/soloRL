import numpy as np

try: 
   import matplotlib
   font = {'family' : 'normal',
           'weight' : 'bold',
           'size'   : 15} 
   matplotlib.rc('font', **font)
   import matplotlib.pyplot as plt 
   import seaborn as sns 
   sns.set(style="darkgrid")
except:
    pass

class Logger:
    def __init__(self, max_len, dt=0.002):

        self.dt = dt
        self.top = 0
        self._axs = [[],[],[],[],[],[]]

        self.actions = np.zeros((max_len, 1))

        self.vel = np.zeros((max_len, 6))
        self.vel_ref = np.zeros((max_len, 6))
        self.torques = np.zeros((max_len, 12))

        self.base_rpy = np.zeros((max_len, 3))
        self.base_xyz = np.zeros((max_len, 3))

        self.power_joints = np.zeros((max_len, 12))

    def log(self, k, vel, vel_ref, torques, joints_power, base_xyz, base_rpy, action):

        self.vel[k] = vel
        self.vel_ref[k] = vel_ref
        self.torques[k] = torques

        self.base_rpy[k] = base_rpy
        self.base_xyz[k] = base_xyz

        self.power_joints[k] = joints_power

        self.actions[k] = action

        self.top += 1

    def reset(self):
        self.vel  *= 0
        self.vel_ref *= 0
        self.torques *= 0
                    
        self.base_rpy *= 0
        self.base_xyz *= 0

        self.power_joints *= 0

        self.actions *= 0
        self.top = 0
    
    def plot_stats(self, label=None):
        if self._axs[0] == []:
            plt.figure(1)
            self._axs[0] = [plt.subplot(3,1,1), plt.subplot(3,1,2), plt.subplot(3,1,3)]
        self.plot_vels(self._axs[0], label) 
        if self._axs[1] == []:
            plt.figure(2)
            self._axs[1] = [plt.subplot(3,1,1), plt.subplot(3,1,2), plt.subplot(3,1,3)]
        self.plot_rpz(self._axs[1], label)
        if self._axs[2] == []:
            plt.figure(3)
            self._axs[2] = [plt.subplot(2,1,1), plt.subplot(2,1,2)]
        self.plot_total_power(self._axs[2], label)
        if self._axs[3] == []:
            plt.figure(4)
            self._axs[3] = [plt.subplot(1,1,1)]
        self.plot_actions(self._axs[3], label)
        return

    def plot_vels(self, axs, label=None):
        '''plot body vel and reference body vel
        '''
        plt.figure(1)
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt

        ax = axs[0]
        #ax.set_xlim(-2, x[-1] + 1000 * self.dt)
        ax.plot(x, self.vel_ref[:self.top,0], linestyle='--', label='Ref Vel', color='black') 
        ax.plot(x, self.vel[:self.top,0], label=label) 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Linear Velocity [m/s]')

        ax = axs[1]
        #ax.set_xlim(-2, x[-1] + 1000 * self.dt)
        ax.plot(x, self.vel_ref[:self.top,1], linestyle='--', label='Ref Vel', color='black') 
        ax.plot(x, self.vel[:self.top,1], label='Ref Vel') 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Lateral Velocity [m/s]')

        ax = axs[2]
        #ax.set_xlim(-2, x[-1] + 1000* self.dt)
        ax.plot(x, self.vel_ref[:self.top,-1], linestyle='--', label='Ref Vel', color='black') 
        ax.plot(x, self.vel[:self.top,-1], label='Ref Vel') 
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Angular Velocity [rad/s]')

        return ax

    def plot_rpz(self, axs, label=None):
        '''plot body vel and reference body vel
        '''
        plt.figure(2)
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt

        ax = axs[0]
        #ax.set_xlim(-2, x[-1] + 1000 * self.dt)
        ax.plot(x, self.base_xyz[:self.top,-1], label=label) 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Height [m]')

        ax = axs[1]
        #ax.set_xlim(-2, x[-1] + 1000 * self.dt)
        ax.plot(x, self.base_rpy[:self.top,0]) 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Roll [rad]')

        ax = axs[2]
        #ax.set_xlim(-2, x[-1] + 1000 * self.dt)
        ax.plot(x, self.base_rpy[:self.top,1]) 
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Pitch [rad]')

        return ax

    def plot_total_power(self, axs, label=None):
        '''plot Total power of the robot and cumulative energy
        '''
        plt.figure(3)
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt
        total_power = self.power_joints[:self.top].sum(1)
        
        joints_energy = self.power_joints[:self.top].cumsum(0) * self.dt
        cumm_energy = joints_energy.sum(1)

        ax = axs[0]
        #ax.set_xlim(-2, x[-1] + 1000*self.dt)
        ax.plot(x, total_power, label=label) 
        ax.set_ylabel('Power Loss [W]')

        ax = axs[1]
        #ax.set_xlim(-2, x[-1] + 1000 * self.dt)
        ax.plot(x, cumm_energy) 
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Energy Loss [J]')

        return ax

    def plot_actions(self, axs, label=None):
        '''plot actions selected
        '''
        plt.figure(4)
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt
        
        ax = axs[0]
        #ax.set_xlim(-2, x[-1] + 1000*self.dt)
        ax.set_ylim(0, 0.72)
        ax.plot(x, self.actions[:self.top], label=label) 
        ax.set_ylabel('Policy Actions')
        ax.set_xlabel('Time [sec]')

        return ax

