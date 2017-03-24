import numpy as np
import os

times = dict()
errs = dict()
surfRegErrs = dict()
for root, dirs, files in os.walk("."):
  r, d = os.path.split(root)
  for f in files:
    if f == "avgFrameTime.csv":
      with open(os.path.join(root,f)) as i:
        times[os.path.split(d)[1]] = float(i.readline()[:-1])
    if f == "trajectoryError.csv":
      with open(os.path.join(root,f)) as i:
        errs[os.path.split(d)[1]] = float(i.readline()[:-1])
    if f == "surfRegErr.csv":
      with open(os.path.join(root,f)) as i:
        surfRegErrs[os.path.split(d)[1]] = float(i.readline()[:-1])
keys = errs.keys()
keys.sort()
for key in keys:
  print key, errs[key], times[key]

keys = surfRegErrs.keys()
keys.sort()
for key in keys:
  print key, surfRegErrs[key]
