import cv2
import os.path, re
import numpy as np

for root, dirs, files in os.walk("./"):
  for f in files:
    if (re.search("capture_?[0-9]*.png",f)):
      print root+f
      I = cv2.imread(os.path.join(root,f),cv2.IMREAD_ANYDEPTH)
      Imin = I.min()
      Imax = I.max()
      print Imin, Imax
      I = (255*(I.astype(float)-Imin)/(Imax-Imin)).astype(np.uint8)
      cv2.imshow("I",I)
      cv2.imwrite(os.path.join(root,os.path.splitext(f)[0]+"_8bit.png"),I)
      cv2.waitKey(100)
  break
