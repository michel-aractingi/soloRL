import pybullet as p
import gym
import numpy as np
from collections import deque
from soloRL.controllers.PD import PD
from soloRL.simulation import SimulatedScene
from pybullet_envs.robot_bases import Joint, BodyPart

CONTACT_DIST=9

class SoloBase:
    """
    Base class for solo robot
    """
    loaded = False

    def __init__(self, 
                 model_urdf, 
                 frame_skip=1, 
                 control='torque', 
                 task='',
                 scene_timestep=1./240.,
                 flat_ground=True,
                 use_treadmill=False,
                 gains=None,
                 gui=False, 
                 num_history_stack=0,
                 self_collision=False):

        self.ordered_joints = []
        self.joints = {}
        self.joints_idx = []
    
        self.feet = []
        self.feet_idx = []

        self.control = control
        self.gains = gains

        self.gui = gui
        self.model_urdf = model_urdf
        self.frame_skip = frame_skip
        self.flat_ground = flat_ground
        self.use_treadmill = use_treadmill
        self.num_history_stack = num_history_stack
        self.self_collision = self_collision
        
        self.state_history = deque(maxlen=self.num_history_stack)
        
        self.scene_timestep = scene_timestep
        self.task = task
        self.initial_z = .35
        self.max_joint_torque = 3
  
    def load(self):
        if self.loaded: 
            return

        self.load_simulation(self.gui)
        self.load_robots()
        self.load_parts([self.id])
        self.load_control_settings()
        self.load_task_settings()
        self.load_history()
        self.load_visualizations()
        p.stepSimulation()
        self.loaded = True

    def load_robots(self):
        if self.loaded: 
            return
        flags = p.URDF_USE_SELF_COLLISION if self.self_collision else 0
        self.id = p.loadURDF(self.model_urdf, flags=flags)

    def load_simulation(self, gui=False):
        if self.loaded: 
            return

        # Start the bullet physics client
        if gui:
            self.phys_id = p.connect(p.GUI)
        else:
            self.phys_id = p.connect(p.DIRECT)
    
        self.scene = SimulatedScene(self.flat_ground, 
                                    use_treadmill=self.use_treadmill,
                                    scene_timestep=self.scene_timestep)
        self.scene.load()
        self.ground_id = self.scene.ground_id

    def load_parts(self, bodies):
        if self.loaded: 
            return

        for i in range(len(bodies)):
            for j in range(p.getNumJoints(bodies[i])):
                joint_info = p.getJointInfo(bodies[i], j) 
                joint_name = joint_info[1].decode("utf8")
                if "ANKLE" not in joint_name:
                    self.joints[joint_name] = Joint(p, joint_name, bodies, i, j)
                    self.ordered_joints.append(self.joints[joint_name])
                    self.joints_idx.append(j)
                else:
                    self.feet.append(Joint(p, joint_name, bodies, i, j))
                    self.feet[-1].disable_motor()
                    self.feet_idx.append(j)
                    
        
        self.joint_state_limit = self.ordered_joints[0].upperLimit
        self.joint_vel_limit = 100

    def load_control_settings(self):
        if self.loaded:
            return

        # disable default motor control
        # recommened by Coumans, having a small positive value instead of 0
        # will mimic joint friction
        # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.jxof6bt5vhut
        
        p.setJointMotorControlArray(self.id,
                                    self.joints_idx,
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[0 for _ in self.joints_idx],
                                    forces=[0.0 for _ in self.joints_idx])
        
        p.setJointMotorControlArray(self.id,
                                    self.joints_idx,
                                    controlMode=p.VELOCITY_CONTROL, 
                                    targetVelocities=[0 for _ in self.joints_idx],
                                    forces=[0.0 for _ in self.joints_idx])
        # set torque control 
        p.setJointMotorControlArray(self.id,
                                    self.joints_idx,
                                    controlMode=p.TORQUE_CONTROL, 
                                    forces=[0.0 for _ in self.joints_idx])

    def load_task_settings(self):
        if self.loaded:
            return
        if self.task == 'pointgoal':
            self.goal_radius = 2.0
            self.goal_xy = np.array([0.0, 0.0])
            self.potential = 0.0
            self.progress = 0.0

    def load_history(self):
        if self.loaded:
            return

        for _ in range(self.num_history_stack):
            self.state_history.append(np.zeros((self.get_current_state().shape[0])))

    def load_visualizations(self):
        if self.loaded or not self.gui or self.task!='pointgoal':
            return

        goal_viz_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.2)
        pose = [*self.goal_xy, 0.0] #self.initial_z]
        self.goal_viz_id = p.createMultiBody(baseMass=0, 
                                             baseCollisionShapeIndex=-1,
                                             baseVisualShapeIndex=goal_viz_id,
                                             basePosition=pose)
        p.changeVisualShape(self.goal_viz_id, -1, rgbaColor=[1,0,0,0.3])

    def reset(self):
        self.robot_specific_reset()
        self.scene.reset()
        
        for _ in range(self.num_history_stack):
            self.state_history.append(self.get_current_state())

        if self.task == 'pointgoal':
            self.sample_goal_point()
            self.goals_reached = 0.0
            self.potential = self.calc_potential()
            self.progress = 0.0

        self.visualization_step()
        state = self.calc_state()
        return state

    def disconnect(self):
        p.disconnect()
 
    def calc_state(self):
        
        # get the state of the robot at this timestep
        current_state = self.get_current_state()
        state = current_state.copy()

        # concatenate history and actions if necessary
        for old_state in reversed(self.state_history):
            state = np.concatenate((state, current_state - old_state))

        return state

    def get_current_state(self):

        # Get body state and velocities
        pos_orn = p.getBasePositionAndOrientation(self.id)
        body_z = np.array([pos_orn[0][-1]])
        body_euler =  np.array([p.getEulerFromQuaternion(pos_orn[1])]).flatten()
        body_vel = np.array(p.getBaseVelocity(self.id)).flatten()

        body_euler = (body_euler % 2*np.pi ) / (2*np.pi) 
        #body_vel = body_vel.clip(-10,10) / 10

        joints_pos, joints_vel = zip(*[j.get_state() for j in self.ordered_joints])
        joints_pos = np.array(joints_pos) / self.joint_state_limit
        joints_vel = np.array(joints_vel) / self.joint_vel_limit

        #joints_at_limit = np.count_nonzero(np.abs(joints_pos) > 0.3)

        feet_contact = self.get_feet_ground_contact()
        state = np.concatenate((body_z, body_euler, body_vel, joints_pos, joints_vel, feet_contact))

        if self.task == 'pointgoal':
            pg_state = self.get_pointgoal_state()
            state = np.concatenate((state, pg_state))

        return state

    def apply_action(self, actions):

        assert len(actions) == len(self.ordered_joints)

        if self.control == 'torque':
            actions = np.clip(actions, -1, 1) * self.max_joint_torque

        elif self.control in ('pd', 'fpd', 'fixed_pd'):
            N_actuators = len(self.ordered_joints)

            q_ref = actions.clip(-1,1) * self.joint_state_limit 

            q, q_dot = zip(*[j.get_state() for j in self.ordered_joints])
            q = np.array(q) 
            q_dot = np.array(q_dot)

            Kp, Kd = self.gains
            actions = PD(q_ref, q, q_dot, Kp, Kd, self.max_joint_torque)

        elif self.control in ('vpd', 'variable_pd'):
            N_actuators = len(self.ordered_joints)

            q_ref = actions[:N_actuators].clip(-1, 1)  * self.joint_state_limit 

            q, q_dot = zip(*np.array([j.get_state() for j in self.ordered_joints]))

            Kp, Kd = actions[-2:]
            actions = PD(q_ref, q, q_dot, Kp, Kd, self.max_joint_torque)

        else: 
            raise NotImplementedError

        p.setJointMotorControlArray(self.id,
                                    self.joints_idx,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=actions)

    def simulator_step(self):
        self.state_history.append(self.get_current_state())

        for _ in range(self.frame_skip):
            p.stepSimulation()

        if self.task == 'pointgoal':
            # check if goal reached and sample new goal
            self.progress = self.calc_progress()
            if self.potential < 0.5:
                self.goals_reached += 1
                self.sample_goal_point()
        
        self.visualization_step()

    def visualization_step(self):
        if not self.gui or self.task!='pointgoal':
            return
        pose = [*self.goal_xy, 0] #self.initial_z]
        p.resetBasePositionAndOrientation(self.goal_viz_id, pose, [0,0,0,1])

    def calc_potential(self):
        robot_xy = self.get_position()[:2]
        return np.linalg.norm(robot_xy - self.goal_xy)

    def calc_progress(self):
        old_potential = self.potential
        self.potential = self.calc_potential()
        return -1 * (self.potential - old_potential)

    def robot_specific_reset(self):
        base_pose=[0,0,self.initial_z]
        base_orientation=[0,0,0,1]
        p.resetBasePositionAndOrientation(self.id, base_pose, base_orientation)

        [p.resetJointState(self.id, j, 0.0) for j in self.joints_idx]

    def get_linear_velocity(self):
        return np.array(p.getBaseVelocity(self.id)[0])

    def get_position(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[0])

    def get_orientation(self):
        quat = p.getBasePositionAndOrientation(self.id)[1]
        euler =  np.array(p.getEulerFromQuaternion(quat))
        #return (euler % 2*np.pi ) / (2*np.pi)
        return euler

    def get_feet_ground_contact(self):
        feet_contact = np.zeros(len(self.feet))
        for i, foot_id in enumerate(self.feet_idx):
            contacts = p.getContactPoints(bodyA=self.ground_id,
                                          bodyB=self.id,
                                          linkIndexA=-1,
                                          linkIndexB=foot_id,
                                          physicsClientId=self.phys_id)
            for c in contacts:
                if c[CONTACT_DIST] < 0.2:
                    feet_contact[i] = 1.0
                    continue

        return feet_contact

    def sample_goal_point(self):
        assert self.task == 'pointgoal'
        xy = np.random.uniform(low=1.0, high=self.goal_radius, size=2)
        sign = np.random.choice([-1.,1.], size=2, replace=True)

        self.goal_xy = sign*xy

    def increment_goal_radius(self, value=1.0):
        assert self.task=='pointgoal'
        self.goal_radius += value


    def get_pointgoal_state(self):
        assert self.task=='pointgoal'
        robot_xy = self.get_position()[:2]
        return np.concatenate((robot_xy, self.goal_xy)) / 2.0

