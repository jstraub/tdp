import numpy as np
import cv2 
import os.path, re

for root, dirs, files in os.walk("./"):
  for f in files:
    if (re.search("capture_stream[0-9]*_img_[0-9]*.png",f)):
      print root+f
      I = cv2.imread(os.path.join(root,f))
      Imin = I.min()
      Imax = I.max()
      print Imin, Imax
      grey = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
      cv2.imshow("I",grey)
      cv2.imwrite(os.path.join(root,os.path.splitext(f)[0]+"_8bit.png"),grey)
      cv2.waitKey(10)
  break
