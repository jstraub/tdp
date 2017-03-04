import numpy as np

xSum = np.zeros(3)
xSqSum = np.zeros((3,3))
N = 0
xVar = []
xVar2 = []
xMean = []

mean = 0
M2  = 0
for i in range(100000):
  x = np.random.randn(3)*100+4
  xSum = (xSum*N + x)/(N+1)
  xSqSum = (xSqSum*N + np.outer(x,x))/(N+1)
  N = min(500, N+1)
  xMean.append(xSum)
  xVar.append(xSqSum - np.outer(xSum,xSum))

  if i > 480 and i < 1020 :
    print i
    print xVar[-1]

import matplotlib.pyplot as plt

plt.figure()
plt.plot(xMean)
plt.plot([m+np.sqrt(xVar[i]) for i,m in enumerate(xMean)])
plt.show()

