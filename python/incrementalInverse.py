import numpy as np
import time
from scipy.linalg import inv, det


print "conditional entropy speed tests"
A = np.random.randn(6,6)
Ainv = inv(A)
t0 = time.time()
for i in range(10000):
  Ai = np.random.randn(6)
  B = np.outer(Ai,Ai)
  H_n = det(A)
  A += B
  H_ni = det(A)
  dH = H_ni - H_n
tE = time.time()
dtNaive = tE - t0
print "dt naive", dtNaive
A = inv(Ainv)
t0 = time.time()
for i in range(10000):
  Ai = np.random.randn(6)
  B = np.outer(Ai,Ai)
#  dH = -0.5*np.log(1. + Ai.dot(Ainv).dot(Ai) )
  dH = Ai.dot(Ainv).dot(Ai)
  A += B
  Ainv -= Ainv.dot(B).dot(Ainv)/(1.+np.trace(B.dot(Ainv)))
tE = time.time()
dtInvInc = tE - t0
print "dt incremental inverse", dtInvInc
print "speedup ", dtNaive/dtInvInc

#http://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices$
#http://www.jstor.org/stable/2690437?seq=1#page_scan_tab_contents
A = np.random.randn(6,6)
#A = np.eye(6)
Ainv = inv(A)

for i in range(100):
  Ai = np.random.randn(6)
  B = np.outer(Ai,Ai)
  
  Ainv -= Ainv.dot(B).dot(Ainv)/(1.+np.trace(B.dot(Ainv)))
  A += B
#  print np.trace((inv(A)-Ainv).dot(inv(A)-Ainv))
  
print "inversion speed tests"
A = np.random.randn(6,6)
#A = np.eye(6)
Ainv = inv(A)
t0 = time.time()
for i in range(10000):
  Ai = np.random.randn(6)
  B = np.outer(Ai,Ai)
  A += B
  invA = inv(A)
tE = time.time()
dtInv = tE - t0
print "dt inverse", dtInv
t0 = time.time()
for i in range(10000):
  Ai = np.random.randn(6)
  B = np.outer(Ai,Ai)
  Ainv -= Ainv.dot(B).dot(Ainv)/(1.+np.trace(B.dot(Ainv)))
tE = time.time()
dtInvInc = tE - t0
print "dt incremental inverse", dtInvInc
print "speedup ", dtInv/dtInvInc
