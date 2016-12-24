import numpy as np

def sq(a):
  return a*a

#class Plane:
#  def __init__(self, p, n):
#    self.p = p
#    self.n = n
#pls = [Plane(np.array([])

def normalize(n):
  return (n.T / np.sqrt((n**2).sum(axis=1))).T

n = np.array([
  [1,0.1,0], 
  [0,1,0], 
  [-1,0,0], 
  [0,-1,0], 
  ])

print normalize(n)

p = np.array([
  [1.1,0,0], 
  [0,1,0], 
  [-1,0,0], 
  [0,-1,0], 
  ])

G = np.array([
  [0 , 1, 0,  1],
  [ 1, 0, 1,  0],
  [0 , 1, 0,  1],
  [ 1, 0, 1,  0],
  ])

p2plObs = np.array([
  [0 ,-1, 0, -1],
  [-1, 0,-1,  0],
  [0 ,-1, 0, -1],
  [-1, 0,-1,  0],
  ])

n2nObs = np.array([
  [ 0, 0, 0,  0],
  [ 0, 0, 0,  0],
  [ 0, 0, 0,  0],
  [ 0, 0, 0,  0],
  ])


def F():
  f = 0
  for i in range(4):
    for j in range(4):
      if G[i,j] > 0:
        f += sq(n[i,:].dot(p[j,:]-p[i,:]) - p2plObs[i,j])
        f += sq(n[i,:].dot(n[j,:]) - n2nObs[i,j])
  return f

def Jn(i):
  J = np.zeros(3)
  for j in range(4):
    if G[i,j] > 0:
      J += 2*(n[i,:].dot(p[j,:]-p[i,:]) - p2plObs[i,j])*(p[j,:]-p[i,:])
      J += 2*(n[i,:].dot(n[j,:]) - n2nObs[i,j])*n[j,:]
  return J

def Jp(i):
  J = np.zeros(3)
  for j in range(4):
    if G[i,j] > 0:
      J += 2*(n[i,:].dot(p[j,:]-p[i,:]) - p2plObs[i,j])*(-n[i,:])
  return J

p0 = np.copy(p)
n0 = np.copy(n)

alpha = 0.1
Jns = np.zeros_like(n)
Jps = np.zeros_like(p)
for it in range(100):
  print F()
  for i in range(4):
    Jns[i,:] = Jn(i)
    Jps[i,:] = Jp(i)
  n = normalize(n - Jns*alpha)
  p = p - Jps*alpha

print p0
print p 
print n0
print n 

