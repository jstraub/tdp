import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from helpers import *
from js.utils.plot.colors import colorScheme

mpl.rc('font',size=30) 
mpl.rc('lines',linewidth=2.)
figSize = (21, 10)
figSize = (16, 10)
cs = colorScheme("label")

pathToTimings = "../results/ablation_fr2_xyz/fullmode/timings.txt"
timings = ParseTimings(pathToTimings)
pathToStats = "../results/ablation_fr2_xyz/fullmode/stats.txt"
stats = ParseTimings(pathToStats)

for key,vals in stats.iteritems():
  print key, len(vals)
for key,vals in timings.iteritems():
  print key, len(vals)

keyToCap = {"FullLoop":"Main Thread", "sampleNormals":"Sample $n$",
    "sampleParams":"Sample $\\mu$, $\\tau$",
    "samplePoints":"Sample $p$",
    "sampleLabels":"Sample $z$",
    "icp": "ICP",
    "icpRGBGPU": "Rot. Pre-align",
    "dataAssoc": "proj. data assoc",
    "extractAssoc": "data assoc filtering",
    "mask": "plane proposals",
    "Setup": "data preproc",
    "inverseIndex": "inv. index",
    "newPlanes": "extract new planes",
    "updatePlanes": "extract plane obs.",
    "trackingMaxStd":"Pose Uncertainty Max Std",
    "trackingMinStd":"Pose Uncertainty Min Std",
    "NumSurfels":"# Map Surfels",
    "NumPlanesInView":"# Surfels In View",
    "NumNewPlanes":"# New Surfels",
    "NumPlanesTracked":"# Surfels Tracked",
    "pTotalNumSample":"# Samples $p$",
    "nTotalNumSample":"# Samples $n$",
    "zTotalNumSample":"# Samples $z$",
    }

keysToPrint = ["mask","icp","FullLoop","Setup","sampleNormals","sampleParams","samplePoints" ]
keysToPrint = ["Draw3D","Draw2D","Setup","sampleNormals","sampleParams","samplePoints" ]
keysToPrint = ["Draw3D","Draw2D","FullLoop"]

keysToPrint = ["trackingH"]
keysToPrint = ["trackingMaxStd","trackingMinStd" ]
fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
i = 0
for key in keysToPrint:
  vals = stats[key]
  if key in keyToCap:
    label = keyToCap[key]
  else:
    label = key
  plt.plot(np.arange(len(vals)), vals, color=cs[i], label=label)
  i = (i+1)%len(cs)
plt.xlim([0,len(vals)])
plt.xlabel("frame")
plt.ylabel("standard deviation")
plt.legend(loc="best")
plt.savefig("statsTracking.png", figure=fig)
plt.close(fig)

keysToPrint = ["NumSurfels","NumPlanesInView", 
#    "NumPlanesProjected",
    "NumNewPlanes", 
#    "NumPruned", 
    "NumPlanesTracked"]
fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
i = 0
for key in keysToPrint:
  vals = stats[key]
  if key in keyToCap:
    label = keyToCap[key]
  else:
    label = key
  plt.plot(np.arange(len(vals)), vals, color=cs[i], label=label)
  i = (i+1)%len(cs)
plt.xlim([0,len(vals)])
plt.xlabel("frame")
plt.ylabel("count")
plt.legend(loc="best")
plt.savefig("statsNumSurfels.png", figure=fig)
plt.close(fig)

keysToPrint = [
#    "pMaxNumSample", "nMaxNumSample", "zMaxNumSample",
    "pTotalNumSample", "nTotalNumSample", "zTotalNumSample"]
keyToCapMeans = {
    "pTotalNumSample":"# Samples $p$ per Surfel",
    "nTotalNumSample":"# Samples $n$ per Surfel",
    "zTotalNumSample":"# Samples $z$ per Surfel"}
numSurfels = np.array(stats["NumSurfels"])
fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
i = 0
for key in keysToPrint:
  vals = stats[key]
  if key in keyToCapMeans:
    label = keyToCapMeans[key]
  else:
    label = key
  plt.plot(np.arange(len(vals)), np.array(vals)/numSurfels, color=cs[i], label=label)
  i = (i+1)%len(cs)
plt.xlim([0,len(vals)])
plt.yscale("log")
plt.xlabel("frame")
plt.ylabel("count")
plt.legend(loc="best")
plt.savefig("statsNumSamples.png", figure=fig)
plt.close(fig)

# high level
keysToPrint = ["FullLoop","sampleNormals","sampleParams","samplePoints","sampleLabels"
    ]
fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
i = 0
for key in keysToPrint:
  vals = timings[key]
  plt.plot(np.arange(len(vals)), vals, color=cs[i],
      label=keyToCap[key])
  i = (i+1)%len(cs)
plt.xlim([0,len(vals)])
plt.xlabel("frame")
plt.ylabel("time [ms]")
plt.legend(loc="best")
plt.savefig("timingsHighLevel.png", figure=fig)
#plt.show()
plt.close(fig)

# main thread
keysToPrint = ["icp","dataAssoc","icpRGBGPU","extractAssoc", "mask" ,
    "Setup", "inverseIndex", "newPlanes", "updatePlanes" ]
icp = np.array(timings["icp"]) + np.array(timings["icpRGBGPU"]) + np.array(timings["inverseIndex"]) 
setup = np.array(timings["Setup"])
assoc = np.array(timings["extractAssoc"]) + np.array(timings["dataAssoc"])
mapUpd = np.array(timings["newPlanes"]) + np.array(timings["updatePlanes"]) + np.array(timings["mask"]) 
  
fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
plt.plot(np.arange(len(icp)), icp, color=cs[0], label="Camera Tracking")
plt.plot(np.arange(len(icp)), setup, color=cs[1], label="Preprocessing")
plt.plot(np.arange(len(icp)), assoc, color=cs[2], label="Data Association")
plt.plot(np.arange(len(icp)), mapUpd, color=cs[3], label="Observation Extraction")
plt.legend(loc="best")
plt.xlim([0,len(vals)])
plt.xlabel("frame")
plt.ylabel("time [ms]")
plt.savefig("timingsMainThread.png", figure=fig)

#plt.show()

