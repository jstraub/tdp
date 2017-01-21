import numpy as np
import matplotlib.pyplot as plt

with open("../build/sparsePlaneSeg_1000.csv","r") as f:
  x = np.loadtxt(f);
  N = x.shape[1]-1
x = (x.T / x[:,-1])[:N,:]
print x.shape

y = np.zeros_like(x)
for i in range(x.shape[1]):
  y[:,i] = 18**2*np.arange(N).astype(np.float)/(640.*480)

plt.figure()
plt.plot(y,x)
plt.ylim([0,1])

c1 = (1,0,0)
c2 = (1,1,0)
c3 = (0,1,0)
c4 = (0,1,1)

plt.figure()
plt.plot(y[:,0], np.mean(x,axis=1), label="mean",color=c1)
plt.plot(y[:,0], np.mean(x,axis=1)+np.std(x,axis=1), label="std",color=c2)
plt.plot(y[:,0], np.mean(x,axis=1)-np.std(x,axis=1), color=c2)
plt.plot(y[:,0], np.median(x,axis=1), label="median",color=c3)
plt.plot(y[:,0], np.percentile(x,95,axis=1), label="95%",color=c4)
plt.plot(y[:,0], np.percentile(x,5,axis=1), label="5%",color=c4)
#plt.boxplot(x.T)
plt.ylim([0,1])

plt.show()
