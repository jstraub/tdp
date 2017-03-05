import numpy as np
import json
import subprocess as subp

pathVarsMapGen = "../build/varsMapGenerated.json"
pathVarsIcpGen = "../build/varsIcpGenerated.json"

def SetMode(samplePoints, dirObsSelect, gradNormObsSelect):
  varsMap = json.load(open("../build/varsMapBase.json","r"))
  if samplePoints:
    varsMap["vars"]["mapPanel.samplePoints"] = "1"
  else:
    varsMap["vars"]["mapPanel.samplePoints"] = "0"
  varsMap["vars"]["mapPanel.save Pc on Finish"] = "1"
  varsMap["vars"]["mapPanel.exit on Finish"] = "1"
  json.dump(varsMap,open(pathVarsMapGen,"w"),indent=4,sort_keys=True)

  varsMap = json.load(open("../build/varsIcpBase.json","r"))
  if dirObsSelect:
    varsMap["vars"]["icpPanel.semObsSelect"] = "1"
  else:
    varsMap["vars"]["icpPanel.semObsSelect"] = "0"
  if gradNormObsSelect:
    varsMap["vars"]["icpPanel.sortByGradient"] = "1"
  else:
    varsMap["vars"]["icpPanel.sortByGradient"] = "0"
  json.dump(varsMap,open(pathVarsIcpGen,"w"),indent=4,sort_keys=True)
def SetToSimpleMap():
  SetMode(samplePoints=False, dirObsSelect=True, gradNormObsSelect=True);
def SetToFullMode():
  SetMode(samplePoints=True, dirObsSelect=True, gradNormObsSelect=True);
def SetToRandomTracking():
  SetMode(samplePoints=True, dirObsSelect=False, gradNormObsSelect=False);
def SetToSimpleMapRandomTracking():
  SetMode(samplePoints=False, dirObsSelect=False, gradNormObsSelect=False);

def Run(dataString, configString, outputPath,
    gtPath):
  args = ["../build/experiments/sparseFusion/sparseFusion",
      dataString, 
      configString,
      pathVarsMapGen,
      pathVarsIcpGen
      ]
  print args.join(" ")
  err = subp.call(args.join(" "), shell=True)
  if err:
    print "error"

  subp.call("mkdir -p "+outputPath, shell=True)
  subp.call("cp timings.txt "+outputPath, shell=True)
  subp.call("cp stats.txt "+outputPath, shell=True)
  subp.call("cp trajectory_tumFormat.csv "+outputPath, shell=True)
  subp.call("cp surfelMap.ply "+outputPath, shell=True)

  subp.call("python evaluate.py trajectory_tumFormat.csv "
      +gtPath+" > " +outputPath"/trajectoryError.csv", shell=True)


datsetPath = "file://"
configString = "config/"
tag = "test"

outputPath = "../results/"+tag+"/fullmode/"
SetToFullMode()
Run(datsetPath, configString, outputPath);

outputPath = "../results/"+tag+"/simpleMap/"
SetToSimpleMap()
Run(datsetPath, configString, outputPath);




