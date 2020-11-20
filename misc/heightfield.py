import pybullet as p
import pybullet_data as pd
import math
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

textureId = -1

import random
random.seed(10)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
heightPerturbationRange = 0.04

numHeightfieldRows = 512
numHeightfieldColumns = 512
heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
for j in range (int(numHeightfieldColumns/2)):
  for i in range (int(numHeightfieldRows/2) ):
    height = random.uniform(0,heightPerturbationRange)
    heightfieldData[2*i+2*j*numHeightfieldRows]=height
    heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
    heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
    heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
      
      
terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
terrain  = p.createMultiBody(0, terrainShape)
p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])

p.changeVisualShape(terrain, -1, rgbaColor=[.76,.69,0.50,1])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.setRealTimeSimulation(1)

import pudb; pudb.set_trace()
while p.isConnected():
    time.sleep(0.01)
