import numpy as np
import json
import time
import subprocess as subp
from helpers import *

configString = "../config/tum_fb2_2017_02_19.json"
for name in ["pioneer_360","desk","xyz","rpy"]:
  datsetPath = "files://../build/rgbd_dataset_freiburg2_"+name+"/[depth,rgb]/*png"
  pathToGt = "../build/rgbd_dataset_freiburg2_"+name+"/groundtruth.txt"
  tag = "fr2_" + name
  outputPath = "../results/"+tag+"/fullmode/"
  SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
  Run(datsetPath, configString, outputPath, pathToGt);

configString = "../config/tum_fb3_2017_02_19.json"
for name in ["long_office_household","nostructure_texture_near_withloop"]:
  datsetPath = "files://../build/rgbd_dataset_freiburg3_"+name+"/[depth,rgb]/*png"
  pathToGt = "../build/rgbd_dataset_freiburg3_"+name+"/groundtruth.txt"
  tag = "fr3_" + name
  outputPath = "../results/"+tag+"/fullmode/"
  SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
  Run(datsetPath, configString, outputPath, pathToGt);

configString = "../config/tum_fb1_2017_02_13.json"
for name in ["desk","desk2","xyz","rpy"]:
  datsetPath = "files://../build/rgbd_dataset_freiburg1_"+name+"/[depth,rgb]/*png"
  pathToGt = "../build/rgbd_dataset_freiburg1_"+name+"/groundtruth.txt"
  tag = "fr1_" + name
  outputPath = "../results/"+tag+"/fullmode/"
  SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
  Run(datsetPath, configString, outputPath, pathToGt);

subp.call("cp -r ../results /home/jstraub/Dropbox/0gtd/thesis/",shell=True)

