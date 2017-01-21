import numpy as np
import matplotlib.pyplot as plt

with open("../build/sparsePlaneSeg_100.csv","r") as f:
  x = np.loadtxt(f);
x = (x.T / x[:,-1])[:100,:]
print x.shape

y = np.zeros_like(x)
for i in range(x.shape[1]):
  y[:,i] = 18**2*np.arange(100).astype(np.float)/(640.*480)

plt.figure()
plt.plot(y,x)
plt.ylim([0,1])

plt.figure()
plt.boxplot(x.T)
plt.ylim([0,1])

plt.show()
