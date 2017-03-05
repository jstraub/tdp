import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from helpers import *
from js.utils.plot.colors import colorScheme

mpl.rc('font',size=30) 
mpl.rc('lines',linewidth=2.)
figSize = (14, 10)
cs = colorScheme("label")

pathToTimings = "../results/ablation_fr2_xyz/fullmode/timings.txt"
timings = ParseTimings(pathToTimings)

for key,vals in timings.iteritems():
  print key, len(vals)

keyToCap = {"FullLoop":"main thread", "sampleNormals":"sample $n$",
    "sampleParams":"sample $\\mu$, $\\tau$",
    "samplePoints":"sample $p$",
    "sampleLabels":"sample $z$",
    "icp": "icp",
    "icpRGBGPU": "rot. pre-align",
    "dataAssoc": "proj. data assoc",
    "extractAssoc": "data assoc filtering",
    "mask": "plane proposals",
    "Setup": "data preproc",
    "inverseIndex": "inv. index",
    "newPlanes": "extract new planes",
    "updatePlanes": "extract plane obs.",
    }

keysToPrint = ["mask","icp","FullLoop","Setup","sampleNormals","sampleParams","samplePoints" ]
keysToPrint = ["Draw3D","Draw2D","Setup","sampleNormals","sampleParams","samplePoints" ]

keysToPrint = ["Draw3D","Draw2D","FullLoop"]

# high level
keysToPrint = ["FullLoop","sampleNormals","sampleParams","samplePoints","sampleLabels" ]

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
i = 0
for key in keysToPrint:
  vals = timings[key]
  plt.plot(np.arange(len(vals)), vals, color=cs[i],
      label=keyToCap[key])
  i+=1
plt.legend(loc="best")
plt.savefig("timingsHighLevel.png", figure=fig)
plt.close(fig)

# main thread
keysToPrint = ["icp","dataAssoc","icpRGBGPU","extractAssoc", "mask" ,
    "Setup", "inverseIndex", "newPlanes", "updatePlanes","FullLoop" ]

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
i = 0
for key in keysToPrint:
  vals = timings[key]
  if key in keyToCap:
    label = keyToCap[key]
  else:
    label = key
  plt.plot(np.arange(len(vals)), vals, color=cs[i], label=label)
  i = (i+1)%len(cs)
plt.legend(loc="best")
plt.savefig("timingsMainThread.png", figure=fig)

plt.show()

