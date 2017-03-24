import numpy as np
import os

times = dict()
errs = dict()
surfRegErrs = dict()
for root, dirs, files in os.walk("."):
  r, d = os.path.split(root)
  if d == "fullmode" or d == "simpleMap":
    if r[:9] == "./surfReg":
      for f in files:
#        if f == "avgFrameTime.csv":
#          with open(os.path.join(root,f)) as i:
#            line =i.readline()[:-1]
#            print line
#            times[os.path.split(d)[1]] = float(line)
        if f == "trajectoryError.csv":
          print root, f
          with open(os.path.join(root,f)) as i:
            line = i.readline()[:-1]
            print line
            errs[os.path.split(d)[1]] = float(line)
        if f == "surfRegErr.csv":
          print root, f
          with open(os.path.join(root,f)) as i:
            line = i.readline()[:-1]
            print line
            surfRegErrs[os.path.split(d)[1]] = float(line)
keys = errs.keys()
keys.sort()
for key in keys:
  print key, errs[key], times[key]

keys = surfRegErrs.keys()
keys.sort()
for key in keys:
  print key, surfRegErrs[key]
