import numpy as np
from scipy.linalg import inv, det, sqrtm, eig, eigh
from scipy.special import ellipe, hyp2f1, i0
from Gaus2SurfHelpers import *
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from js.geometry.icosphere import *

theta = 30.*np.pi/180.
R = np.array([[1., 0., 0.], 
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]])

L = np.eye(3)
L[0,0] = 100.
L[1,1] = 100.
L[2,2] = 10.
S = R.T.dot(L).dot(R)
Ldiag = np.diag(L).tolist()
print Ldiag

nLvls = 4
KLxs = np.zeros((nLvls, 2))
KLns = np.zeros((nLvls, 2))

Nr = 100000
for lvl in range(nLvls):
  x = sqrtm(S).dot(np.random.randn(3,Nr))
  ns = np.zeros((x.shape[0],x.shape[1]/2))
  for i in range(0,Nr,2):
    ns[:,i/2] = np.cross(x[:,i],x[:,i+1])
  # flip normals randomly to get bimodal distribution
  sign = np.sign(np.random.randn(ns.shape[1]))
  while (sign == 0.).sum() > 0:
    sign[sign==0] = np.sign(np.random.randn((sign == 0.).sum())) 
  ns *= sign

  print "using icosahedron-based histogram"
  sHistX = SphereHistogram(level=lvl)
  sHistX.Compute(x.T)
  ds = sHistX.GetTriangleCenters()
  pdfXico = DirectionalPdfInDirs(ds.T,S)
  HppX = sHistX.Entropy(x.T)
  HpqX = sHistX.CrossEntropy(x.T, pdfXico)
  print "Hpp(x): ",HppX
  print "Hpq(x): ",HpqX
  print "KL: ",HpqX - HppX
  KLxs[lvl,1] = HpqX - HppX
  #sHist.PlotHist()

  sHistN = SphereHistogram(level=lvl)
  sHistN.Compute(ns.T)
  ds = sHistN.GetTriangleCenters()
  pdfNico = SurfPdfInDirs(ds.T,S)
  HppN = sHistN.Entropy(ns.T)
  HpqN = sHistN.CrossEntropy(ns.T, pdfNico)
  print "Hpp(n): ",HppN
  print "Hpq(n): ",HpqN
  print "KL: ",HpqN - HppN
  KLns[lvl,1] = HpqN - HppN

  N = int(np.floor(np.sqrt(sHistN.GetNumTriangles())))

  print "using angle-based histogram"
  pdfX = DirectionalPdf(S,N)
  pdfN = SurfPdf(S,N)
  pX = angleHist(x,N)
  KLx2 = KL(x, pX, pdfX, N)
  KLxs[lvl,0] = KLx2
  print "KL x: ", KLx2
  #KLx = np.sum(pX[pX>0]*(np.log(pX[pX>0]) - np.log(pdfX[pX>0])))
  #print "KL x: ", KLx

  pN = angleHist(ns,N)
  KLn2 = KL(ns, pN, pdfN, N)
  print "KL n: ", KLn2
  KLns[lvl,0] = KLn2
  #KLn = np.sum(pdfN[pN>0]*(np.log(pdfN[pN>0]) - np.log(pN[pN>0])))
  #print "KL n: ", KLn


np.savetxt("KLxs.csv", KLxs)
np.savetxt("KLns.csv", KLns)

fig = plt.figure()
plt.title("KL x")
plt.plot(KLxs[:,0], label="angle")
plt.plot(KLxs[:,1], label="ico")
plt.legend(loc="best")

fig = plt.figure()
plt.title("KL n")
plt.plot(KLns[:,0], label="angle")
plt.plot(KLns[:,1], label="ico")
plt.legend(loc="best")

plt.show()
