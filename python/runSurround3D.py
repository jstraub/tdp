import numpy as np
import os.path
import re
import subprocess as subp
import time

def runSurround3D(path,logFile):
  args = [os.path.abspath("../build/experiments/surround3D/surround3D"),
    "file://"+os.path.abspath(path)+"/",
    os.path.abspath("../config/surround3D_2016_09_13.json") ]
  print ' '.join(args)
  print ' --------------------- '
  err = subp.call(' '.join(args),shell=True)
  now = time.strftime("%Y-%m-%d %H:%M")
  if err:
    print 'error when executing ', ' '.join(args)
    with open(logFile,"a") as f:
      f.write(now+': ERROR when executing '+' '.join(args)+"\n");
  else:
    with open(logFile,"a") as f:
      f.write(now+': executed '+' '.join(args)+"\n");
def runIcpSamFusion(path,logFile):
  args = [os.path.abspath("../build/experiments/icpSamFusion/icpSamFusion"),
    "file://"+os.path.abspath(path)+"/",
    os.path.abspath("../config/surround3D_2016_10_06.json") ,
    '-1' ]
  print ' '.join(args)
  print ' --------------------- '
  err = subp.call(' '.join(args),shell=True)
  now = time.strftime("%Y-%m-%d %H:%M")
  if err:
    print 'error when executing ', ' '.join(args)
    with open(logFile,"a") as f:
      f.write(now+': ERROR when executing '+' '.join(args)+"\n");
  else:
    with open(logFile,"a") as f:
      f.write(now+': executed '+' '.join(args)+"\n");

def runMarchingCubes(path,logFile):
  args = [os.path.abspath("../build/experiments/marchingCubes/marchingCubes"),
    os.path.abspath(path)+"/tsdf.raw",
    "-1"]
  print ' '.join(args)
  print ' --------------------- '
  err = subp.call(' '.join(args),shell=True)
  now = time.strftime("%Y-%m-%d %H:%M")
  if err:
    print 'error when executing ', ' '.join(args)
    with open(logFile,"a") as f:
      f.write(now+': ERROR when executing '+' '.join(args)+"\n");
  else:
    with open(logFile,"a") as f:
      f.write(now+': executed '+' '.join(args)+"\n");

path = "../build/"
paths = []
for root, dirs, files in os.walk(path):
  for d in dirs:
    if os.path.isfile(os.path.join(root+d,"video.pango")) \
        and os.path.isfile(os.path.join(root+d,"imu.pango")) :
      if not re.search("32-", d) is None:
        print "found dir", root+d
        paths.append(os.path.join(root,d))

doRunSurround3D = True
doRunMarchingCubes = True

if doRunSurround3D:
  for path in paths:
#    runSurround3D(path, "./log_runSurround3D.txt")
    runIcpSamFusion(path, "./log_runIcpSamFusion.txt")

if doRunMarchingCubes:
  for path in paths:
    runMarchingCubes(path, "./log_runSurround3D.txt")
