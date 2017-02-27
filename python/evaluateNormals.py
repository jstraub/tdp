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
    pdf = np.exp((mu.dot(ds)-1)*tau) * tau / (2.*np.pi*(1.-np.exp(-2.*tau)))
  return pdf

def logvMF(ds,mu, tau):
  if tau == 0.:
    pdf = -np.ones(ds.size/3)*np.log(4.*np.pi)
  else:
    pdf = (mu.dot(ds)-1)*tau + np.log(tau) -np.log(2.*np.pi) -np.log(1.-np.exp(-2.*tau))
  return pdf

def logKL(x, pX, logpdfX, N):
  Hpp = 0.
  Hpq = 0.
  Ni = x.shape[1]
  thetas = np.linspace(0.+90./N,180.-90./N,N)
  for i in range(Ni):
    phi = np.arctan2(x[1,i],x[0,i])*180./np.pi
    theta = np.arctan2(np.sqrt(x[0,i]**2+x[1,i]**2),x[2,i])*180./np.pi 
    iphi = int(np.floor((N-1)*(phi+180.)/(360.)))
    itheta = int(np.floor((N-1)*(theta)/(180.)))
#    w = np.sin(thetas[itheta]*np.pi/180.)*(np.pi/N)*(2*np.pi/N)
#    print pX[itheta,iphi]/w
    Hpq -= logpdfX[itheta,iphi]
#    Hpp -= pX[itheta,iphi] * np.log(pX[itheta,iphi]/w)
    Hpp -= np.log(pX[itheta,iphi]) #/w)
  Hpq /= Ni
  Hpp /= Ni
  print " "
  print "Hpq ", Hpq
  print "Hpp ", Hpp
  return Hpq - Hpp

def logvMFpdfGrid(N,mu,tau):
  pdfN = np.zeros((N-1, N-1))
  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)
  for i,alpha in enumerate(alphas):
    for j,beta in enumerate(betas):
      theta = (alpha)*np.pi/180.
      phi = (beta)*np.pi/180.
      n = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),np.sin(theta)])
      pdfN[i,j] = logvMF(n,mu, tau)
  return pdfN

def norm(x):
  return np.sqrt((x**2).sum())



lvl = 4
#N = int(np.floor(np.sqrt(sHistN.GetNumTriangles())))
#N = 1000
plt.figure()
for path in ["./normals_600_voting.csv", "./normals_1000_voting.csv",
    "./normals_1300_voting.csv"]:

  with open(path) as f:
    print f.readline()
    print f.readline()
    ns = np.loadtxt(f)
  print ns.shape
  ns = ns[:min(ns.shape[0],10000),:3]

  # improve conditioning of grid based method
  #R = np.array([[0,0,1],
  #              [1,0,0],
  #              [0,1,0]])
  #print det(R)
  #ns = (R.dot(ns.T)).T

  print ns.shape
  nSum = ns.sum(axis=0) 
  mu = nSum/norm(nSum)
  #mu = np.array([0,0,-1])

  sHistN = SphereHistogram(level=lvl)
  sHistN.Compute(ns)
  ds = sHistN.GetTriangleCenters()
  HppN = sHistN.Entropy(ns)
  #pN = angleHist(ns.T,N)
  #taus = [0, 5, 10, 12.5, 15, 17.5, 20, 25, 35]
  taus = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
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
  #  print "using angle-based histogram"
  #  logpdfN = logvMFpdfGrid(N,mu,tau)
  #  print logpdfN
  #  KLn2 = logKL(ns.T, pN, logpdfN, N)
  #  print "KL n: ", KLn2
  #  KLs2.append(KLn2)


  plt.plot(taus, KLs, label=path)
  #plt.plot(taus, KLs2, label="angle hist")
plt.legend(loc="best")
plt.show()
