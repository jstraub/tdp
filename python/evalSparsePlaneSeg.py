import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from js.utils.plot.colors import colorScheme

c1 = colorScheme("labelMap")["turquoise"]
c2 = colorScheme("labelMap")["orange"]
c3 = colorScheme("labelMap")["blue"]
c4 = colorScheme("labelMap")["red"]
mpl.rc('font',size=32) 
mpl.rc('lines',linewidth=3.)
figSize = (14, 12)

#with open("./sparsePlaneSeg_NYU_1000.csv","r") as f:
#with open("./sparsePlaneSeg_1000.csv","r") as f:
with open("./sparsePlaneSeg_LS_1000.csv","r") as f:
  x = np.loadtxt(f);
xObs = x[:,1::2]
x = x[:,0::2]
N = x.shape[1]-1
numObs = xObs[:,-1]
xObs = (xObs.T / xObs[:,-1])[:N,:]
x = (x.T / x[:,-1])[:N,:]
print x.shape
print xObs.shape

y = np.zeros_like(x)
for i in range(x.shape[1]):
  y[:,i] = 18**2*np.arange(N).astype(np.float)/(640.*480)

xShow = np.linspace(0.,0.7,100)
xIn = np.zeros(1002)
yIn = np.zeros(1002)
xIn[-1], yIn[-1] = 1., 1.
yInterp = np.zeros((100,N))

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
for i in range(N):
  xIn[1:-1] = xObs[:,i]
  yIn[1:-1] = x[:,i]
  f = interp1d(xIn,yIn) #,kind="cubic")
  yInterp[:,i] = f(xShow)

plt.plot(xShow,np.mean(yInterp,axis=1), label="mean",color=c1)
plt.fill_between(xShow, np.mean(yInterp,axis=1)-np.std(yInterp,axis=1),\
    np.mean(yInterp,axis=1)+np.std(yInterp,axis=1),color=c2,alpha=0.3)
plt.plot(xShow, np.median(yInterp,axis=1), label="median",color=c3)
plt.plot(xShow, np.percentile(yInterp,95,axis=1), label="95%",color=c4)
plt.plot(xShow, np.percentile(yInterp,5,axis=1), label="5%",color=c4)
plt.ylim([0,1])
plt.xlim([0,0.7])
plt.xlabel("% of used data")
plt.ylabel("% of inlier data")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("inliersVsPercentUsedData.png", figure=fig)

xShow = np.linspace(0.,999,100)
#xShow = np.linspace(0.,1000./numObs.max(),100)
fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
for i in range(N):
#  xIn[1:-1] = xObs[:,i]
  xIn[1:-1] = np.arange(1000).astype(np.float)#/numObs[i]
  yIn[1:-1] = x[:,i]
  f = interp1d(xIn,yIn) #,kind="cubic")
  yInterp[:,i] = f(xShow)

plt.plot(xShow,np.mean(yInterp,axis=1), label="mean",color=c1)
plt.fill_between(xShow, np.mean(yInterp,axis=1)-np.std(yInterp,axis=1),\
    np.mean(yInterp,axis=1)+np.std(yInterp,axis=1),color=c2,alpha=0.3)
plt.plot(xShow, np.median(yInterp,axis=1), label="median",color=c3)
plt.plot(xShow, np.percentile(yInterp,95,axis=1), label="95%",color=c4)
plt.plot(xShow, np.percentile(yInterp,5,axis=1), label="5%",color=c4)
plt.ylim([0,1])
#plt.xlim([0,0.01])
plt.xlabel("# planes")
plt.ylabel("% of inlier data")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("inliersVsNumPlanes.png", figure=fig)

#plt.figure()
#plt.plot(y[:,0], np.mean(x,axis=1), label="mean",color=c1)
#plt.plot(y[:,0], np.mean(x,axis=1)+np.std(x,axis=1), label="std",color=c2)
#plt.plot(y[:,0], np.mean(x,axis=1)-np.std(x,axis=1), color=c2)
#plt.fill_between(y[:,0], np.mean(x,axis=1)-np.std(x,axis=1),\
#    np.mean(x,axis=1)+np.std(x,axis=1),color=c2,alpha=0.3)
#plt.plot(y[:,0], np.median(x,axis=1), label="median",color=c3)
#plt.plot(y[:,0], np.percentile(x,95,axis=1), label="95%",color=c4)
#plt.plot(y[:,0], np.percentile(x,5,axis=1), label="5%",color=c4)
##plt.boxplot(x.T)
#plt.ylim([0,1])
#plt.xlim([0,1])
#plt.legend(loc="best")

plt.show()
