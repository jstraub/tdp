import os.path, re
import os
import subprocess as subp

for root,dirs,files in os.walk("./"):
  for f in files:
    print "convert {} -quality 100 {}".format(f,re.sub("png","jpg",f))
    subp.call("convert {} -quality 100 {}".format(f,re.sub("png","jpg",f)),shell=True) 
  break
