import numpy as np
import matplotlib.pyplot as plt

pathToTimings = "/home/jstraub/Dropbox/0gtd/thesis/timings.txt"

frameId = []
timings = dict()
with open(pathToTimings) as f:
  lines = f.readlines()
  for line in lines:
    if line[:5] == "Frame":
      desc, num = line[:-1].split(" ")
      frameId.append(int(num))
    else:
      desc, num = line[:-1].split("\t")
      if frameId[-1] > 1:
        if desc in timings.keys():
          timings[desc].append(float(num))
        else:
          timings[desc] = [float(num)]

for key,vals in timings.iteritems():
  print key, len(vals)

keysToPrint = ["Draw3D","Draw2D","Setup","sampleNormals","sampleParams","samplePoints" ]
keysToPrint = ["mask","icp","FullLoop","Setup","sampleNormals","sampleParams","samplePoints" ]

fig = plt.figure()
for key,vals in timings.iteritems():
  if key in keysToPrint:
    plt.plot(np.arange(len(vals)), vals, label=key)
plt.legend()
plt.show()
