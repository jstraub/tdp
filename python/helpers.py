import numpy as np


def ParseTimings(pathToTimings):
  frameId = []
  timings = dict()
  with open(pathToTimings) as f:
    lines = f.readlines()
    for line in lines:
      desc, num = line[:-1].split("\t")
      if desc == "Frame":
        frameId.append(int(num))
      else:
        if frameId[-1] > 1:
          if desc in timings.keys():
            timings[desc].append(float(num))
          else:
            timings[desc] = [float(num)]
  return timings
