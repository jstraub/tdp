import numpy as np

xSum = 0
N = 0
xSums = []
for i in range(10000):
  x = np.random.randn(1)*10+4
  xSum = (xSum*N/(N+1) + x)
  N = min(500, N+1)
  xSums.append(xSum/N)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(xSums)
plt.show()

