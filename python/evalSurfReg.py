import numpy as np
import json
import time
import subprocess as subp
from helpers import *

configString = "../config/iclnuim_2017_02_23.json"
for i in range(4):
  datsetPath = "files://../build/living_room_traj{}n_frei_png/[depth,rgb]/*png".format(i)
  pathToGt = "../build/living_room_traj{}n_frei_png/livingRoom{}n.gt.freiburg".format(i,i)
  tag = "surfReg_kt{}n".format(i)
  outputPath = "../results/"+tag+"/fullmode/"
  SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
  Run(datsetPath, configString, outputPath, pathToGt);

  outputPath = "../results/"+tag+"/simpleMap/"
  SetMode(samplePoints=False, dirObsSelect=True, gradNormObsSelect=True);
  Run(datsetPath, configString, outputPath, pathToGt);

  # prints out the registration error
  args = ["./../../3rdparty/SurfReg/src/build/SurfReg",
      "-r "+outputPath+"/surfelMap.ply",
      "-m ../build/living-room.ply"]
  subp.call(" ".join(args) +" > " +outputPath+"/surfRegErr.csv", shell=True)
