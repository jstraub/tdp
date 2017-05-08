import numpy as np
import os

def GetATE(path):
  with open(os.path.join(path,"trajectoryError.csv")) as i:
    errStr = i.readline()[:-1]
    if len(errStr)>0:
      err = float(errStr)
    else:
      err = np.nan
  return err

errsDir = dict()
errsRnd = dict()
for name in ["traj0n","traj1n","traj2n","traj3n"]:
  errsDir[name] = []
  errsRnd[name] = []
for name in ["desk","xyz"]:
  errsDir[name] = []
  errsRnd[name] = []

for i in range(10):
  for name in ["traj0n","traj1n","traj2n","traj3n"]:
    tag = "iclnuim_" + name + "_{}".format(i)
    outputPath = "../results/"+tag+"/fullmode/"
    if os.path.isdir(outputPath):
      errsDir[name].append(GetATE(outputPath))
    outputPath = "../results/"+tag+"/simpleMapRndTrack//"
    if os.path.isdir(outputPath):
      errsRnd[name].append(GetATE(outputPath))

  for name in ["desk","xyz"]:
    tag = "fr2_" + name + "_{}".format(i)
    outputPath = "../results/"+tag+"/fullmode/"
    if os.path.isdir(outputPath):
      errsDir[name].append(GetATE(outputPath))
    outputPath = "../results/"+tag+"/simpleMapRndTrack//"
    if os.path.isdir(outputPath):
      errsRnd[name].append(GetATE(outputPath))


keys = errsDir.keys()
keys.sort()
for key in keys:
  errsDir[key] = [ err for err in errsDir[key] if not np.isnan(err)]
  print key, errsDir[key]

keys = errsRnd.keys()
keys.sort()
for key in keys:
  errsRnd[key] = [ err for err in errsRnd[key] if not np.isnan(err)]
  print key, errsRnd[key]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from js.utils.plot.colors import colorScheme

mpl.rc('font',size=30) 
mpl.rc('lines',linewidth=2.)
figSize = (16, 10)
figSize = (21, 10)
cs = colorScheme("label") 

data = [ 'kt0' , 'kt1' , 'kt2' , 'kt3' , 'fr2_xyz' , 'fr2_desk' ]
DVOSLAM = [ 0.032 ,0.061,0.119 ,0.053  , 0.021 , 0.017 ]
RGBDSLAM = [ 0.044 ,0.032,0.031 ,0.167  , 0.023 , 0.095  ]
ElasticFusion = [ 0.009 , 0.009 , 0.014 , 0.106 , 0.011 , np.nan ]
Kintinuous = [ 0.072 , 0.005 , 0.010 , 0.355 , 0.029 , 0.034]
DensePlanarSLAM =[ 0.246 , 0.0169 , np.nan , np.nan , np.nan , np.nan]
CPASLAM = [ np.nan , np.nan , np.nan , np.nan , 0.014 , 0.046 ]
#DirFusion =[ 0.005 ,  0.007 , 0.017  , 0.417 , 0.015 , 0.066]
DirFusion =[ np.mean(errsDir["traj0n"]) ,  
    np.mean(errsDir["traj1n"]),
    np.mean(errsDir["traj2n"]),
    np.mean(errsDir["traj3n"]),
    np.mean(errsDir["xyz"]),
    np.mean(errsDir["desk"])]
DirFusionRnd =[ np.mean(errsRnd["traj0n"]) ,  
    np.mean(errsRnd["traj1n"]),
    np.mean(errsRnd["traj2n"]),
    np.mean(errsRnd["traj3n"]),
    np.mean(errsRnd["xyz"]),
    np.mean(errsRnd["desk"])]

DirFusionStd =[ np.std(errsDir["traj0n"]) ,  
    np.std(errsDir["traj1n"]),
    np.std(errsDir["traj2n"]),
    np.std(errsDir["traj3n"]),
    np.std(errsDir["xyz"]),
    np.std(errsDir["desk"])]
DirFusionRndStd =[ np.std(errsRnd["traj0n"]) ,  
    np.std(errsRnd["traj1n"]),
    np.std(errsRnd["traj2n"]),
    np.std(errsRnd["traj3n"]),
    np.std(errsRnd["xyz"]),
    np.std(errsRnd["desk"])]

data = [ 'kt0' , 'kt1' , 'kt2' , 'fr2_xyz' , 'fr2_desk' ]
DVOSLAM = [ 0.032 ,0.061,0.119 , 0.021 , 0.017 ]
RGBDSLAM = [ 0.044 ,0.032,0.031 , 0.023 , 0.095  ]
ElasticFusion = [ 0.009 , 0.009 , 0.014 , 0.011 , np.nan ]
Kintinuous = [ 0.072 , 0.005 , 0.010 ,  0.029 , 0.034]
DensePlanarSLAM =[ 0.246 , 0.0169 , np.nan ,  np.nan , np.nan]
CPASLAM = [ np.nan , np.nan , np.nan ,  0.014 , 0.046 ]
#DirFusion =[ 0.005 ,  0.007 , 0.017  , 0.417 , 0.015 , 0.066]
DirFusion =[ np.mean(errsDir["traj0n"]) ,  
    np.mean(errsDir["traj1n"]),
    np.mean(errsDir["traj2n"]),
    np.mean(errsDir["xyz"]),
    np.mean(errsDir["desk"])]
DirFusionRnd =[ np.mean(errsRnd["traj0n"]) ,  
    np.mean(errsRnd["traj1n"]),
    np.mean(errsRnd["traj2n"]),
    np.mean(errsRnd["xyz"]),
    np.mean(errsRnd["desk"])]

DirFusionStd =[ np.std(errsDir["traj0n"]) ,  
    np.std(errsDir["traj1n"]),
    np.std(errsDir["traj2n"]),
    np.std(errsDir["xyz"]),
    np.std(errsDir["desk"])]
DirFusionRndStd =[ np.std(errsRnd["traj0n"]) ,  
    np.std(errsRnd["traj1n"]),
    np.std(errsRnd["traj2n"]),
    np.std(errsRnd["xyz"]),
    np.std(errsRnd["desk"])]

ind = np.arange(len(data))
width=0.125

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
ax = plt.subplot(111)

ax.bar(ind, DVOSLAM, width,color=cs[6], label="DVO-SLAM")
ax.bar(ind+width,   RGBDSLAM, width, color=cs[0],label="RGBD-SLAM")
ax.bar(ind+2*width, ElasticFusion, width, color=cs[1],label="Elastic Fusion")
ax.bar(ind+3*width, Kintinuous, width, color=cs[2],label="Kintinuous")
ax.bar(ind+4*width, DensePlanarSLAM, width, color=cs[3],label="Dense Planar SLAM")
ax.bar(ind+5*width, CPASLAM, width, color=cs[4],label="CPA-SLAM")

ax.bar(ind+6*width, DirFusion, width, yerr=DirFusionStd,color=cs[5],label="Dir.-SLAM")
#ax.bar(ind+7*width, DirFusionRnd, width,  yerr=DirFusionRndStd,color=cs[6],label="Dir.-SLAM Rnd")

ax.set_ylabel("Absolute Trajectory Error [m]")
ax.set_xticks(ind+3.5*width)
ax.set_xticklabels(data)
plt.xlim([0,8])
plt.legend(loc="best")

plt.savefig("./ate.png",fig=fig)
plt.show()

