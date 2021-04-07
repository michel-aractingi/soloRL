import numpy as np
import pybullet as p
import pybullet_data as pd

class SimulatedScene:
    loaded = False
    def __init__(self, flat_ground, use_treadmill=False, scene_timestep=None, nogravity=False):
        self.flat_ground = flat_ground
        self.use_treadmill = use_treadmill if flat_ground else False
        self.nogravity = nogravity
        self.scene_timestep = scene_timestep 
    
    def load(self):
        if self.loaded:
            return
        p.setAdditionalSearchPath(pd.getDataPath())
        
        if not self.nogravity:
            p.setGravity(0, 0, -9.81)
        
        if self.flat_ground:
            # load plane ground
            self.ground_id = p.loadURDF("plane.urdf")
            if self.use_treadmill:
                self.treadmill = Treadmill()
                self.treadmill.load()
        else:
            self.ground_heightfield = Heightfield(maxheight=0.04)
            self.ground_heightfield.load()
            self.ground_id = self.ground_heightfield.terrain_id

        # set timestep (LAAS default 0.001)
        p.setTimeStep(self.scene_timestep)

        self.loaded = True

    def reset(self):
        if self.flat_ground:
            if self.use_treadmill:
                self.treadmill.reset()
            return
        # comment out for now due to speed
        #self.ground_heightfield.update_randomly()

class Treadmill:
    loaded=False
    def __init__(self, base_vel=None, length=50, max_vel=2):
        self.max_vel = max_vel
        self.y_shift = 0.49 * np.random.choice([-1,1]) 
        self.length = length
        self.random_vel = lambda: (np.random.random() - 0.5) * 2 * self.max_vel
        self.base_vel = base_vel if base_vel is not None else self.random_vel()


    def load(self):
        if self.loaded:
            return 
        
        self.coll_id = p.createCollisionShape(
                                           p.GEOM_HEIGHTFIELD, 
                                           heightfieldData=np.zeros(self.length*2), 
                                           numHeightfieldRows=self.length, numHeightfieldColumns=2)
        self.body_id = p.createMultiBody(0, self.coll_id) 
        p.changeVisualShape(self.body_id, -1, rgbaColor=[0.4,0.8,0.8,1])
        p.resetBasePositionAndOrientation(self.body_id,
                                          posObj=[0, self.y_shift, 0],
                                          ornObj=[0,0,0,1])
        p.resetBaseVelocity(self.coll_id, [self.base_vel, 0, 0]) 
        self.loaded = True

    def reset(self, randomize_vel=True):
        p.resetBasePositionAndOrientation(self.body_id,
                                          posObj=[0, self.y_shift * np.random.choice([-1,1]), 0],
                                          ornObj=[0,0,0,1])
        if randomize_vel:
            self.base_vel = self.random_vel()
            p.resetBaseVelocity(self.body_id, [self.base_vel, 0, 0])

class Generalfield:
    loaded=False
    def __init__(self, stepwidth, maxheight, maxrows=512, maxcols=512):
        '''
        stepwidth between 10,50
        stepheight around 0.0.5,0.1,0.2
        '''
        self.maxheight = maxheight
        self.maxrows = maxrows
        self.maxcols = maxcols
        self.stepwidth = stepwidth
        self.heightfieldData = np.zeros((self.maxrows, self.maxcols))

    def load(self):
        if self.loaded:
            return

        self.generate_heightfield()
        self.terrain_id = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, 
                                                 meshScale=[.05,.05,1], 
                                                 heightfieldTextureScaling=(self.maxrows-1)/2, 
                                                 heightfieldData=self.heightfieldData.flatten(), 
                                                 numHeightfieldRows=self.maxrows,
                                                 numHeightfieldColumns=self.maxcols)

        terrain  = p.createMultiBody(0, self.terrain_id)
        p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
        p.setAdditionalSearchPath(pd.getDataPath())
        textureid = p.loadTexture("heightmaps/ground0.txt")

        p.changeVisualShape(terrain, -1, 
                            rgbaColor=[0.44705882, 0.51764706, 0.53333333,1],
                            textureUniqueId = textureid)
        self.loaded = True

    def update(self):
        assert self.loaded
        self.generate_heightfield()
        self.terrain_id = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, 
                                                 meshScale=[.05,.05,1], 
                                                 heightfieldTextureScaling=(self.maxrows-1)/2, 
                                                 heightfieldData=self.heightfieldData.flatten(), 
                                                 numHeightfieldRows=self.maxrows,
                                                 numHeightfieldColumns=self.maxcols,
                                                 replaceHeightfieldIndex = self.terrain_id)

    def generate_heightfield(self):
        raise NotImplementedError

class Heightfield(Generalfield):
    def generate_heightfield(self):
        for j in range (int(self.maxcols/2)):
          for i in range (int(self.maxrows/2)):
            height = np.random.uniform(0,self.maxheight)
            self.heightfieldData[2*i+2*j*self.maxrows]=height
            self.heightfieldData[2*i+1+2*j*self.maxrows]=height
            self.heightfieldData[2*i+(2*j+1)*self.maxrows]=height
            self.heightfieldData[2*i+1+(2*j+1)*self.maxrows]=height

class Tiltedfield(Generalfield):
    def generate_heightfield(self):
        for i in range(int(self.maxrows)):
            self.heightfieldData[i] = i
        self.heightfieldData = (self.heightfieldData / self.maxrows) * self.maxheight

class Stairsfield(Generalfield):
    def generate_heightfield(self):
        for i in np.arange(0,self.maxcols,self.stepwidth):
            self.heightfieldData[i:i+self.stepwidth] = (i / self.stepwidth) * self.maxheight
        return 

class Stepfield(Generalfield):
    def generate_heightfield(self):
        for i in np.arange(0,self.maxcols,self.stepwidth):
            for j in np.arange(0,self.maxrows, self.stepwidth):
                self.heightfieldData[i:i+self.stepwidth, j:j+self.stepwidth] = np.random.uniform(0,self.maxheight)

