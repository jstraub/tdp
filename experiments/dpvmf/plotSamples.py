import numpy as np


import mayavi.mlab as mlab

mlab.figure(bgcolor=(1,1,1))
n = np.loadtxt("./vmfSamples_tau10.csv")
mlab.points3d(n[:,0],n[:,1],n[:,2],color=(1,0,0),mode="point")
n = np.loadtxt("./vmfSamples_tau100.csv")
mlab.points3d(n[:,0],n[:,1],n[:,2],color=(0,1,0),mode="point")
n = np.loadtxt("./vmfSamples_tau1000.csv")
mlab.points3d(n[:,0],n[:,1],n[:,2],color=(0,0,1),mode="point")
mlab.show()

