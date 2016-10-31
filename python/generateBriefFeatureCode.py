import numpy as np
import matplotlib.pyplot as plt

# http://www.robots.ox.ac.uk/~vgg/rg/papers/CalonderLSF10.pdf
# sigma=2 and kernel size of 9x9 for Gaussian smoothing

numBits = 256 
numBytes = numBits/8
patchS = 32

def samplePair():
  return np.floor(np.random.randn(2,2) * patchS / 5. + 0.5*(patchS-1.)).astype(int)

# using the random sampling strategy with isotropic Gaussian
n = 0
xs = np.zeros((2,2*numBits),dtype=int)
while n != numBits:
  x = samplePair()
  while np.any(x < 0) or np.any(x >= patchS):
    x = samplePair()
  xs[:,2*n:2*(n+1)] = x
  n += 1

with open("brief.h",'w') as f:
  k = 0
  for i in range(numBytes):
    f.write("brief[{}] = ((img({},{}) < img({},{}) ? {} : 0)\n".format(i,xs[0,k*2],xs[1,k*2],xs[0,k*2+1],xs[1,k*2+1],1))
    k+=1
    for j in range(1,7):
      f.write(" | (img({},{}) < img({},{}) ? {} : 0)\n".format(xs[0,k*2],xs[1,k*2],xs[0,k*2+1],xs[1,k*2+1],2**j))
      k+=1
    f.write(" | (img({},{}) < img({},{}) ? {} : 0));\n".format(xs[0,k*2],xs[1,k*2],xs[0,k*2+1],xs[1,k*2+1],128))

plt.figure()
for i in range(numBits):
  plt.plot(xs[0,i*2:(i+1)*2],xs[1,i*2:(i+1)*2],'r-')
plt.show()
  
