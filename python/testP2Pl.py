import numpy as np
from scipy.linalg import eig, inv

scale = 0.1
N = 100
x = np.zeros((3,N))
x[:2,:] = scale*np.random.randn(2,N)
x[2,:]  = scale*np.random.randn(1,N)*0.01

S = x.dot(x.T)
e,Q = eig(S)
e = e.real
print "cov"
#print S
print e
print e[0]/e.sum(), e[0]/(e[1]+e[2])
#print Q

I = inv(S)
e,Q = eig(I)
e = e.real
print "info"
#print I
print e
#print Q
print e[0]/e.sum(), e[0]/(e[1]+e[2])
#tau = e[0]/(e[1]+e[2])
#tau = e[0]/min(e[1],e[2])
tau = e[0]
mu = Q[:,0]

N = 1000
angs = np.linspace(-np.pi, np.pi,N)
n = np.zeros((3,N))
for i,ang in enumerate(angs):
  n[0,i] = 0.
  n[1,i] = np.cos(ang)
  n[2,i] = np.sin(ang)

logpnSn = np.zeros(N)
for i,ang in enumerate(angs):
  logpnSn[i] = -n[:,i].dot(S).dot(n[:,i])
logpnb = mu.dot(n)*tau

def expDist(logp):
  logZ = np.log(np.sum(np.exp(logp-logp.max())))+logp.max()
  p = np.exp(logp - logZ)
  return p

pnSn = expDist(logpnSn)
pnb = 0.5*expDist(logpnb)

print pnSn.shape
print pnb.shape

import matplotlib.pyplot as plt

plt.figure()
plt.plot(angs,pnSn,label="nSn")
plt.plot(angs,pnb ,label="nb")
plt.legend()
plt.show()



