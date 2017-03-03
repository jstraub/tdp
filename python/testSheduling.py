import numpy as np
import matplotlib.pyplot as plt

def ComputeH(p):
  return - np.sum(p*np.log(p))

I = 100
Ni = np.arange(I).astype(np.float)
alpha = 100


Ha = []
pUnif = 0.9
ps = 1.-(Ni+alpha)/float(Ni.sum()+alpha)
for it in range(300):
  for i in range(I):
    u = np.random.rand(1)
    if u < ps[i]:
      Ni[i] += 1
      ps = 1.-(Ni+alpha)/float(Ni.sum()+alpha)
      Ha.append(ComputeH(Ni/Ni.sum()))
#  print Ni


Ni = np.arange(I).astype(np.float)
Hb = []
pUnif = 0.99
Nmax = Ni.max()
ps = pUnif + (1.-pUnif)*(1.-Ni/float(Nmax))
for it in range(300):
  for i in range(I):
    u = np.random.rand(1)
    if u < ps[i]:
      Ni[i] += 1
      Nmax = Ni.max()
      ps = pUnif + (1.-pUnif)*(1.-Ni/float(Nmax))
#      Hb.append(ComputeH(ps))
      Hb.append(ComputeH(Ni/Ni.sum()))
#      print Ni

plt.figure()
plt.plot(Ha,label="A")
plt.plot(Hb,label="B")
plt.legend()
plt.show()


#for it in range(100):
#  Nmax = Ni.max()
#  cdf = 1.-Ni/Nmax

