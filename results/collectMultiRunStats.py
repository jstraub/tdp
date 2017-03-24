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
