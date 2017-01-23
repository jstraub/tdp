import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from js.utils.plot.colors import colorScheme

mpl.rc('font',size=40) 
mpl.rc('lines',linewidth=4.)
figSize = (14, 12)

with open("./sparsePlaneSeg_NYU_1000.csv","r") as f:
#with open("../build/sparsePlaneSeg_1000.csv","r") as f:
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

c1 = colorScheme("labelMap")["turquoise"]
c2 = colorScheme("labelMap")["orange"]
c3 = colorScheme("labelMap")["blue"]
c4 = colorScheme("labelMap")["red"]


plt.figure()
plt.plot(y[:,0], np.mean(x,axis=1), label="mean",color=c1)
plt.plot(y[:,0], np.mean(x,axis=1)+np.std(x,axis=1), label="std",color=c2)
plt.plot(y[:,0], np.mean(x,axis=1)-np.std(x,axis=1), color=c2)
plt.fill_between(y[:,0], np.mean(x,axis=1)-np.std(x,axis=1),\
    np.mean(x,axis=1)+np.std(x,axis=1),color=c2,alpha=0.3)
plt.plot(y[:,0], np.median(x,axis=1), label="median",color=c3)
plt.plot(y[:,0], np.percentile(x,95,axis=1), label="95%",color=c4)
plt.plot(y[:,0], np.percentile(x,5,axis=1), label="5%",color=c4)
#plt.boxplot(x.T)
plt.ylim([0,1])
plt.xlim([0,1])
plt.legend(loc="best")
plt.show()
