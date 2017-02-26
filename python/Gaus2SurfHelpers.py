import numpy as np
from scipy.linalg import inv, det, eigh, solve
from scipy.special import ellipe, hyp2f1, i0

def EllipticalE(kSquare):
  ''' 
  Elliptical Integal of the second kind can be expressed via 2F1
  '''
  E = np.pi/2.*hyp2f1(0.5,-0.5,1.,kSquare)
  return E

def SurfPdfVal(theta,phi,S):
  D = np.zeros((3,2))
  D[:,0] = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])
  D[:,1] = np.array([-np.sin(phi),np.cos(phi),0.])
  Stilde = D.T.dot(solve(S,D))
  E, U = eigh(inv(Stilde))
  A,C=E.real
  E = EllipticalE(1.-C/A)
  #TODO: Whi pi**3 and not pi**2 ?
  return np.sin(theta)*E**2*C*A**2/(np.pi**3*det(S))
  #return np.sin(theta)*E**2/(np.pi**2*det(S)*A**2*C)
  #return E**2/(np.pi**2*det(S)*A**2*C)
def SurfPdfInDirs(ds,S):
  '''
  compute surface normal pdf in the given directions ds
  '''
  pdf = np.zeros(ds.shape[1])
  phis = np.arctan2(ds[1,:],ds[0,:])
  thetas = np.arctan2(np.sqrt(ds[0,:]**2+ds[1,:]**2),ds[2,:])
  for i in range(ds.shape[1]):
    pdf[i] = SurfPdfVal(thetas[i], phis[i], S)
  return pdf
def SurfPdf(S,N, check=True):
  pdfN = np.zeros((N-1, N-1))
#  alphas = np.linspace(0.,180,N)
#  betas = np.linspace(-180.,180,N)
  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)
  for i,alpha in enumerate(alphas):
    for j,beta in enumerate(betas):
      pdfN[i,j] = SurfPdfVal((alpha)*np.pi/180.,(beta)*np.pi/180.,S);
  # multiply by surface element
#  pdfN *= (np.pi*2./N)*(np.pi/N)
#  pdfN /= pdfN.sum()
#  pdfN /= np.pi
  if check:
    P = 0.
    for i,alpha in enumerate(alphas):
      for j,beta in enumerate(betas):
        # Surface element
        #dS = np.sin(alpha*np.pi/180.)*(np.pi/N)*(2*np.pi/N)
        dS = 2*np.pi/N*np.pi/N
        P+=pdfN[i,j]*dS
    print "integral over SurfPdf: ", P

  return pdfN

def DirectionalPdfValInDir(d,S):
  theta = np.arctan2(np.sqrt(d[0]**2+d[1]**2),d[2])
  return np.sin(theta)/((d.T.dot(inv(S)).dot(d)**(1.5))*np.pi*4.*np.sqrt(det(S)))
def DirectionalPdfVal(theta,phi,S):
  d = np.array([[np.sin(theta)*np.cos(phi)],[np.sin(theta)*np.sin(phi)],[np.cos(theta)]])
  return np.sin(theta)/(((d.T.dot(solve(S,d)))**(1.5))*np.pi*4.*np.sqrt(det(S)))
  #return 1./((d.T.dot(inv(S)).dot(d)**(1.5))*np.pi*4.*np.sqrt(det(S)))
def DirectionalPdfInDirs(ds,S):
  '''
  compute surface normal pdf in the given directions ds
  '''
  pdf = np.zeros(ds.shape[1])
  phis = np.arctan2(ds[1,:],ds[0,:])
  thetas = np.arctan2(np.sqrt(ds[0,:]**2+ds[1,:]**2),ds[2,:])
  for i in range(ds.shape[1]):
    pdf[i] = DirectionalPdfVal(thetas[i], phis[i], S)
  return pdf
def DirectionalPdf(S,N, check=True):
  pdfX = np.zeros((N-1, N-1))
#  alphas = np.linspace(0.,180,N)
#  betas = np.linspace(-180.,180,N)
  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)
  for i,alpha in enumerate(alphas):
    for j,beta in enumerate(betas):
      pdfX[i,j] = DirectionalPdfVal((alpha)*np.pi/180., (beta)*np.pi/180., S)
  # multiply by surface element
#  pdfX *= (np.pi*2./N)*(np.pi/N)
  if check:
    P = 0.
    for i,alpha in enumerate(alphas):
      for j,beta in enumerate(betas):
        # Surface element
        #/dS = np.sin(alpha*np.pi/180.)*(np.pi/N)*(2*np.pi/N)
        #dS = 2*np.sin(np.pi/N)*np.sin(alpha*np.pi/180.)*(2*np.pi/N)
        dS = 2*np.pi/N*np.pi/N
        P+=pdfX[i,j]*dS
    print "integral over DirectionalPdf: ", P
  return pdfX

def angleHist(n,N):
  phis = np.arctan2(n[1,:],n[0,:])*180./np.pi
  thetas = np.arctan2(np.sqrt(n[0,:]**2+n[1,:]**2),n[2,:])*180./np.pi 
  hist = np.zeros((N-1,N-1))
  Ni = n.shape[1]
  for i in range(Ni):
    iphi = int(np.floor((N-1)*(phis[i]+180.)/(360.)))
    itheta = int(np.floor((N-1)*(thetas[i])/(180.)))
    hist[itheta, iphi] += 1
  hist /= np.sum(hist)
  return hist

def binWeights(N):
  w = np.zeros((N-1,N-1))
  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)
  for i,alpha in enumerate(alphas):
    for j,beta in enumerate(betas):
      #hist[i,j] /= (-np.cos(alpha*np.pi/180.+np.pi/N)+np.cos(alpha*np.pi/180.))*(2*np.pi/N)
      #w[i,j] = 2*np.sin(np.pi/N)*np.sin(alpha*np.pi/180.)*(2*np.pi/N)
      w[i,j] = (2*np.pi/N)*np.pi/N
      #hist[i,j] /= np.sin(alpha*np.pi/180.)*(np.pi/N)*(2*np.pi/N)
  return w

def EntropyOfHist(pX,w,N):
  H = 0
  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)
  for i,alpha in enumerate(alphas):
    for j,beta in enumerate(betas):
      H -= np.nan_to_num(pX[i,j]*w[i,j]*np.log(pX[i,j]))
  return H

def CrossEntropy(x, pdfX, N):
  Hpq = 0.
  Ni = x.shape[1]
  thetas = np.linspace(0.+90./N,180.-90./N,N)
  for i in range(Ni):
    phi = np.arctan2(x[1,i],x[0,i])*180./np.pi
    theta = np.arctan2(np.sqrt(x[0,i]**2+x[1,i]**2),x[2,i])*180./np.pi 
    iphi = int(np.floor((N-1)*(phi+180.)/(360.)))
    itheta = int(np.floor((N-1)*(theta)/(180.)))
    Hpq -= np.log(pdfX[itheta,iphi]) 
  Hpq /= Ni
  return Hpq 

def KL(x, pX, pdfX, N):
  Hpp = 0.
  Hpq = 0.
  Ni = x.shape[1]
  thetas = np.linspace(0.+90./N,180.-90./N,N)
  for i in range(Ni):
    phi = np.arctan2(x[1,i],x[0,i])*180./np.pi
    theta = np.arctan2(np.sqrt(x[0,i]**2+x[1,i]**2),x[2,i])*180./np.pi 
    iphi = int(np.floor((N-1)*(phi+180.)/(360.)))
    itheta = int(np.floor((N-1)*(theta)/(180.)))
#    w = np.sin(thetas[itheta]*np.pi/180.)*(np.pi/N)*(2*np.pi/N)
#    print pX[itheta,iphi]/w
    Hpq -= np.log(pdfX[itheta,iphi]) 
    print Hpq
#    Hpp -= pX[itheta,iphi] * np.log(pX[itheta,iphi]/w)
    Hpp -= np.log(pX[itheta,iphi]) #/w)
  Hpq /= Ni
  Hpp /= Ni
  print " "
  print "Hpq ", Hpq
  print "Hpp ", Hpp
  return Hpq - Hpp

if __name__ == "__main__":

  N = 10

  alphas = np.linspace(0.+90./N,180.-90./N,N-1)
  betas = np.linspace(-180.+180./N,180.-180./N,N-1)

  alphas = np.linspace(0.,180.-180./N,N-1)
  betas = np.linspace(-180.,180.-360./N,N-1)

  alphas = np.linspace(0.+180./N,180.-1e-6,N-1)
  betas = np.linspace(-180.+360./N,180.-1e-6,N-1)

  print alphas
  print betas

  thetas = alphas
  phis = betas
  hist = np.zeros((N-1,N-1))
  for i in range(N-1):
    iphi = int(np.floor((N-1)*(phis[i]+180.)/(360.)))
    itheta = int(np.floor((N-1)*(thetas[i])/(180.)))
    print iphi, itheta
    hist[itheta, iphi] += 1

  print hist

