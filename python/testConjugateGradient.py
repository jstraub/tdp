import numpy as np
from scipy.linalg import solve, inv

def solveCG(ATA,ATb,x0):
  x = np.copy(x0)
  r = ATb-ATA.dot(x)
  p = np.copy(r)
  rsold = r.T.dot(r)
  for i in range(ATb.size):
    Ap = ATA.dot(p)
    alpha = rsold/(p.T.dot(Ap))
    x += alpha*p
    r -= alpha*Ap
    rsnew = r.T.dot(r)
    if np.sqrt(rsnew) < 1e-10:
      print i
      break
    p = r+(rsnew/rsold)*p
    rsold = rsnew
  return x

def solveCG_direct(A,b,x0):
  x = np.copy(x0)
  r = A.T.dot(b-A.dot(x))
  p = np.copy(r)
  rsold = r.T.dot(r)
  for i in range(b.size):
    Ap = A.T.dot(A.dot(p))
    alpha = rsold/(p.T.dot(Ap))
    x += alpha*p
    r -= alpha*Ap
    rsnew = r.T.dot(r)
    if np.sqrt(rsnew) < 1e-10:
      print i
      break
    p = r+(rsnew/rsold)*p
    rsold = rsnew
  return x

def solvePCG_direct(A,b,x0,M):
  x = np.copy(x0)
  r = A.T.dot(b-A.dot(x))
  z = solve(M,r)
  p = np.copy(z)
  rsold = r.T.dot(z)
  for i in range(1000):
    Ap = A.T.dot(A.dot(p))
    alpha = rsold/(p.T.dot(Ap))
    x += alpha*p
    r -= alpha*Ap
    if np.sqrt(r.T.dot(r)) < 1e-10:
      print i
      break
    z = solve(M,r)
    rsnew = r.T.dot(z)
    p = z+(rsnew/rsold)*p
    rsold = rsnew
  return x

A = np.random.randn(100,12)
b = np.random.randn(100,1)

ATA = A.T.dot(A)
ATb = A.T.dot(b)

x = solve(ATA,ATb)
print x.T

x = solveCG(ATA,ATb,np.zeros((12,1)))
print x.T

x = solveCG_direct(A,b,np.zeros((12,1)))
print x.T

print "PCG diag(I)"
x = solvePCG_direct(A,b,np.zeros((12,1)),np.eye(12))
print x.T

print "PCG diag(ATA)"
M = np.diag(np.diag(ATA))
x = solvePCG_direct(A,b,np.zeros((12,1)),M)
print x.T

print "PCG ATA"
M = ATA
x = solvePCG_direct(A,b,np.zeros((12,1)),M)
print x.T
