import numpy as np
import json
import time
import subprocess as subp
from helpers import *

datsetPath = "files://../build/rgbd_dataset_freiburg2_xyz/[depth,rgb]/*png"
configString = "../config/tum_fb2_2017_02_19.json"
pathToGt = "../build/rgbd_dataset_freiburg2_xyz/groundtruth.txt"
tag = "ablation_fr2_xyz"

outputPath = "../results/"+tag+"/fullmode/"
SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
Run(datsetPath, configString, outputPath, pathToGt);

outputPath = "../results/"+tag+"/simpleMap/"
SetMode(samplePoints=False, dirObsSelect=True, gradNormObsSelect=True);
Run(datsetPath, configString, outputPath, pathToGt);

outputPath = "../results/"+tag+"/rndTrack/"
SetMode(samplePoints=True, dirObsSelect=False, gradNormObsSelect=False);
Run(datsetPath, configString, outputPath, pathToGt);

outputPath = "../results/"+tag+"/simpleMapRndTrack/"
SetMode(samplePoints=False, dirObsSelect=False, gradNormObsSelect=False);
Run(datsetPath, configString, outputPath, pathToGt);

outputPath = "../results/"+tag+"/dirSegTrackSampleMap/"
SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=False);
Run(datsetPath, configString, outputPath, pathToGt);

outputPath = "../results/"+tag+"/gradNormTrackSampleMap/"
SetMode(samplePoints=True, dirObsSelect=False, gradNormObsSelect=True);
Run(datsetPath, configString, outputPath, pathToGt);

#
