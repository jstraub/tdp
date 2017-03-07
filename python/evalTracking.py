import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from helpers import *
from js.utils.plot.colors import colorScheme

mpl.rc('font',size=30) 
mpl.rc('lines',linewidth=2.)
figSize = (21, 10)
figSize = (16, 10)
cs = colorScheme("label")

#pathToTimings = "../results/ablation_fr2_xyz/fullmode/timings.txt"
#timings = ParseTimings(pathToTimings)
pathToStats = "/home/jstraub/Dropbox/0gtd/thesis/results/ablation_fr2_xyz/fullmode/stats.txt"
statsAwa = ParseTimings(pathToStats)
pathToStats = "/home/jstraub/Dropbox/0gtd/thesis/results/ablation_fr2_xyz/rndTrack/stats.txt"
statsRnd = ParseTimings(pathToStats)

for key,vals in statsAwa.iteritems():
  print key, len(vals)

for key,vals in statsRnd.iteritems():
  print key, len(vals)

N = len(statsRnd["NumPlanesTracked"])

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
plt.plot(np.arange(N), 
    statsRnd["NumPlanesTracked"], label="# tracked by random ICP", color=cs[1])
plt.plot(np.arange(N), 
    statsAwa["NumPlanesTracked"], label="# tracked by dir.-aware ICP", color=cs[0])
plt.plot(np.arange(N),
    statsAwa["NumPlanesInView"], label="# total in view", color=cs[2])
#plt.plot(np.arange(len(statsRnd["NumPlanesTracked"])),
#    statsRnd["NumPlanesInView"], label="random", color=cs[1])
plt.xlim([0,N])
plt.legend(loc="best")
plt.xlabel("frame")
plt.ylabel("number of surfels")
plt.savefig("trackingStratCompPlanesTracked.png", figure=fig)

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
plt.plot(np.arange(N), 
    statsRnd["trackingMaxStd"], label="min/max std of random ICP", color=cs[1])
plt.plot(np.arange(N), 
    statsAwa["trackingMaxStd"], label="min/max std of dir.-aware ICP", color=cs[0])
plt.plot(np.arange(N), 
    statsRnd["trackingMinStd"], color=cs[1])
plt.plot(np.arange(N), 
    statsAwa["trackingMinStd"], color=cs[0])
plt.legend(loc="upper left")
plt.xlim([0,N])
plt.xlabel("frame")
plt.ylabel("standard deviation of pose estimate")
plt.savefig("trackingStratCompStd.png", figure=fig)

fig = plt.figure(figsize = figSize, dpi = 80, facecolor="w", edgecolor="k")
plt.plot(np.arange(N), 
    statsRnd["trackingH"], label="entropy of random ICP", color=cs[1])
plt.plot(np.arange(N), 
    statsAwa["trackingH"], label="entropy of dir.-aware ICP", color=cs[0])
plt.xlim([0,N])
plt.legend(loc="best")
plt.xlabel("frame")
plt.ylabel("entropy of pose estimate")
plt.savefig("trackingStratCompEntropy.png", figure=fig)
plt.show()
