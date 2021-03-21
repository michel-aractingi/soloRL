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

        self.vel = np.zeros((max_len, 6))
        self.vel_ref = np.zeros((max_len, 6))
        self.torques = np.zeros((max_len, 12))

        self.base_rpy = np.zeros((max_len, 3))
        self.base_xyz = np.zeros((max_len, 3))

        self.power_joints = np.zeros((max_len, 12))

    def log(self, k, vel, vel_ref, torques, joints_power, base_xyz, base_rpy):

        self.vel[k] = vel
        self.vel_ref[k] = vel_ref
        self.torques[k] = torques

        self.base_rpy[k] = base_rpy
        self.base_xyz[k] = base_xyz

        self.power_joints[k] = joints_power

        self.top += 1

    def reset(self):
        self.vel  *= 0
        self.vel_ref *= 0
        self.torques *= 0
                    
        self.base_rpy *= 0
        self.base_xyz *= 0

        self.power_joints *= 0

        self.top = 0
    
    def plot_stats(self):
        self.plot_vels() 
        self.plot_rpz()
        self.plot_total_power()
        return

    def plot_vels(self):
        '''plot body vel and reference body vel
        '''
        plt.figure()
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt

        ax = plt.subplot(311)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, self.vel_ref[:self.top,0], linestyle='--', label='Ref Vel', color='black') 
        plt.plot(x, self.vel[:self.top,0], label='Ref Vel') 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Linear Velocity [m/s]')

        ax = plt.subplot(312)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, self.vel_ref[:self.top,1], linestyle='--', label='Ref Vel', color='black') 
        plt.plot(x, self.vel[:self.top,1], label='Ref Vel') 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Lateral Velocity [m/s]')

        ax = plt.subplot(313)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, self.vel_ref[:self.top,-1], linestyle='--', label='Ref Vel', color='black') 
        plt.plot(x, self.vel[:self.top,-1], label='Ref Vel') 
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Angular Velocity [rad/s]')

        return ax

    def plot_rpz(self):
        '''plot body vel and reference body vel
        '''
        plt.figure()
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt

        ax = plt.subplot(311)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, self.base_xyz[:self.top,-1]) 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Height [m]')

        ax = plt.subplot(312)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, self.base_rpy[:self.top,0]) 
        #ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Roll [rad]')

        ax = plt.subplot(313)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, self.base_rpy[:self.top,1]) 
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Pitch [rad]')

        return ax

    def plot_total_power(self):
        '''plot Total power of the robot and cumulative energy
        '''
        plt.figure()
        x = np.arange(0, self.vel.shape[0])[:self.top] * self.dt
        total_power = self.power_joints[:self.top].sum(1)
        
        joints_energy = self.power_joints[:self.top].cumsum(0) * self.dt
        cumm_energy = joints_energy.sum(1)

        ax = plt.subplot(211)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, total_power) 
        ax.set_ylabel('Power Loss [W]')

        ax = plt.subplot(212)
        ax.set_xlim(-3, self.vel.shape[0] * self.dt)
        plt.plot(x, cumm_energy) 
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Energy Loss [J]')

        return ax

