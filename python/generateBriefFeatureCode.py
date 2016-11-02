import numpy as np
import matplotlib.pyplot as plt

# http://www.robots.ox.ac.uk/~vgg/rg/papers/CalonderLSF10.pdf
# sigma=2 and kernel size of 9x9 for Gaussian smoothing

numBits = 256 
numBytes = numBits/8
numInts = numBytes/4
patchS = 32

def samplePair():
  return np.random.randn(2,2) * patchS / 5.

# using the random sampling strategy with isotropic Gaussian
n = 0
xs = np.zeros((2,2*numBits))
while n != numBits:
  x = samplePair()
  while np.any(np.sqrt((x**2).sum(axis=0)) > (patchS-1)*0.5):
    x = samplePair()
  xs[:,2*n:2*(n+1)] = x 
  n += 1

xs0 = np.copy(xs)
with open("brief.h",'w') as f:
  for r in range(0,30):
    f.write("case {}:\n".format(r))
    f.write("  ExtractBrief{}(patch, desc);\n".format(r))
    f.write("  return true;\n".format(r))

  plt.figure()
  for r in range(0,30):
    alpha = r*12./180.*np.pi
    R = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
    xs = np.copy(xs0)
    xs = R.dot(xs)
    xs += 0.5*(patchS-1.)
    xs = np.floor(xs).astype(int)

    if np.any(xs > 31):
      print r,"large values"
    if np.any(xs<0):
      print r,"small values"

    print np.mean(xs,axis=1)

    if True:
      plt.subplot(6,6,r+1)
      for i in range(numBits):
        plt.plot(xs[0,i*2:(i+1)*2],xs[1,i*2:(i+1)*2],'r-')

      plt.xlim([0,31])
      plt.ylim([0,31])

      f.write("void ExtractBrief{}".format(r)+"(const Image<uint8_t>& patch, Vector8uda& desc) {\n")
      k = 0
      for i in range(numInts):
        f.write("desc({}) = ((patch({},{}) < patch({},{}) ? {} : 0)\n".format(i,xs[0,k*2],xs[1,k*2],xs[0,k*2+1],xs[1,k*2+1],1))
        k+=1
        for j in range(1,31):
          f.write(" | (patch({},{}) < patch({},{}) ? {} : 0)\n".format(xs[0,k*2],xs[1,k*2],xs[0,k*2+1],xs[1,k*2+1],2**j))
          k+=1
        f.write(" | (patch({},{}) < patch({},{}) ? {} : 0));\n".format(xs[0,k*2],xs[1,k*2],xs[0,k*2+1],xs[1,k*2+1],2**31))
      f.write("}\n\n")

  plt.show()
  
