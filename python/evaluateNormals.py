import numpy as np
from scipy.linalg import inv, det, sqrtm, eig, eigh
from scipy.special import ellipe, hyp2f1, i0
from Gaus2SurfHelpers import *
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from js.geometry.icosphere import *

def vMF(ds,mu, tau):
  if tau == 0.:
    pdf = np.ones(ds.size/3)*1./(4.*np.pi)
  else:
    pdf = np.exp((mu.dot(ds)-1)*tau) * tau / (2.*np.pi*(1.-np.exp(-tau)))
  return pdf

def vMFpdfGrid(N,mu,tau):
  pdfN = np.zeros((N-1, N-1))
  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)
  for i,alpha in enumerate(alphas):
    for j,beta in enumerate(betas):
      theta = (alpha)*np.pi/180.
      phi = (beta)*np.pi/180.
      n = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),np.sin(theta)])
      pdfN[i,j] = vMF(n,mu, tau)
  return pdfN

def norm(x):
  return np.sqrt((x**2).sum())

with open("./normals_1000.csv") as f:
  print f.readline()
  ns = np.loadtxt(f)

lvl = 4

print ns.shape
nSum = ns.sum(axis=0) 
mu = nSum/norm(nSum)
#mu = np.array([0,0,-1])

sHistN = SphereHistogram(level=lvl)

N = int(np.floor(np.sqrt(sHistN.GetNumTriangles())))

sHistN.Compute(ns)
pN = angleHist(ns,N)
ds = sHistN.GetTriangleCenters()
HppN = sHistN.Entropy(ns)
taus = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
KLs = []
KLs2 = []
for i,tau in enumerate(taus):
  print "using icosahedron-based histogram"
  pdfNico = vMF(ds.T, mu, tau)
  HpqN = sHistN.CrossEntropy(ns, pdfNico)
  print "Hpp(n): ",HppN
  print "Hpq(n): ",HpqN
  print "KL: ",HpqN - HppN
  KLs.append(HpqN - HppN)


  print "using angle-based histogram"
  pdfN = vMFpdfGrid(N,mu,tau)
  KLn2 = KL(ns, pN, pdfN, N)
  print "KL n: ", KLn2
  KLs2.append(KLn2)


plt.figure()
plt.plot(taus, KLs, label="geogrid")
plt.plot(taus, KLs2, label="angle hist")
plt.legend(loc="best")
plt.show()
