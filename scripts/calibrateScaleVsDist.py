#!/usr/bin/env python

import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

mpl.rc('font',size=25)                                                          
mpl.rc('lines',linewidth=4.)

parser = argparse.ArgumentParser(description = 'plots 1D csv files')       
parser.add_argument('-i', help='path to input csv file')
args = parser.parse_args()

sd = np.loadtxt(args.i)
s = sd[:,0]
d = sd[:,1]

A = np.concatenate((d[:,np.newaxis],np.ones_like(d)[:,np.newaxis]),axis=1)
b = s

ATA = A.T.dot(A)
ATb = A.T.dot(b)

x = solve(ATA,ATb)
print x

xi = np.linspace(0.5,3,100)
psi = x[0]*xi + x[1]

plt.figure()
plt.plot(d,s)
plt.plot(xi,psi)

plt.title(args.i)
plt.show()
