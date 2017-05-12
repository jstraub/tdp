import numpy as np
import json
import time
import subprocess as subp
from helpers import *

for i in range(10):
  configString = "../config/iclnuim_2017_02_23.json"
  for name in ["traj0n","traj1n","traj2n","traj3n"]:
    datsetPath = "files://../build/living_room_"+name+"_frei_png/[depth,rgb]/*png"
    pathToGt = "../build/living_room_"+name+"_frei_png/groundtruth.txt"
    tag = "iclnuim_" + name + "_{}".format(i)
    outputPath = "../results/"+tag+"/fullmode/"
    SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
    Run(datsetPath, configString, outputPath, pathToGt);

    outputPath = "../results/"+tag+"/simpleMapRndTrack//"
    SetMode(samplePoints=False, dirObsSelect=False, gradNormObsSelect=False);
    Run(datsetPath, configString, outputPath, pathToGt);

  configString = "../config/tum_fb2_2017_02_19.json"
  for name in ["desk","xyz"]:
    datsetPath = "files://../build/rgbd_dataset_freiburg2_"+name+"/[depth,rgb]/*png"
    pathToGt = "../build/rgbd_dataset_freiburg2_"+name+"/groundtruth.txt"
    tag = "fr2_" + name + "_{}".format(i)
    outputPath = "../results/"+tag+"/fullmode/"
    SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
    Run(datsetPath, configString, outputPath, pathToGt);

    outputPath = "../results/"+tag+"/simpleMapRndTrack/"
    SetMode(samplePoints=False, dirObsSelect=False, gradNormObsSelect=False);
    Run(datsetPath, configString, outputPath, pathToGt);

subp.call("cp -r ../results /home/jstraub/Dropbox/0gtd/cvpr2018/",shell=True)

