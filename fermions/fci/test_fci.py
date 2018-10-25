import fsbasic
import numpy

def genSpinOrbIntegrals(h1e,h2e):
   k0 = h1e.shape[0]
   k = 2*k0 
   h1 = numpy.zeros((k,k))
   h1[0::2,0::2] = h1e
   h1[1::2,1::2] = h1e
   # 2e
   h2 = numpy.zeros((k,k,k,k))
   h2[0::2,0::2,0::2,0::2] = h2e
   h2[1::2,1::2,1::2,1::2] = h2e
   h2[0::2,0::2,1::2,1::2] = h2e
   h2[1::2,1::2,0::2,0::2] = h2e
   # <ij|kl>
   h2a = h2 - h2.transpose(0,3,2,1) 
   h2a = h2a.transpose(0,2,1,3)
   # Unique representation: V[ijkl] (i<j,k<l) = -<ij||kl>
   h2 = numpy.zeros(h2a.shape)
   for j in range(k):
      for i in range(j):
         for l in range(k):
	    for m in range(l):								      
	       h2[i,j,m,l] = -h2a[i,j,m,l]
   return h1,h2

t = -1.0
U = +1.5
k0 = 4
k = 2*k0
ndim = 2**k
E0 = 0.0
H1 = numpy.zeros((k0,k0))
#   1---3
#  /   /
# 0---2
H1[0,2] = H1[2,0] = t 
H1[0,1] = H1[1,0] = t 
H1[1,3] = H1[3,1] = t 
H1[2,3] = H1[3,2] = t 
#for i in range(k0-1):
#    H1[i,i+1] = H1[i+1,i] = -1.0
#H1[k0-1,0] = H1[0,k0-1] = -1.0 # PBC
V2 = numpy.zeros((k0,k0,k0,k0))
for i in range(k0):
   V2[i,i,i,i] = U
H1,V2 = genSpinOrbIntegrals(H1,V2)

import scipy.linalg
e,v = scipy.linalg.eigh(H1)
print 'e=',e

info = [k,ndim,E0,H1,V2]
iop = 0 
if iop == 0:
   efci,civec = fsbasic.fsFCI(info,nroot=8,crit_vec=1.e-4)
   numpy.save('fcivec',civec)
else:
   civec = numpy.load('fcivec.npy')
   neig = civec.shape[0]
   for i in range(neig):
      rdm1 = fsbasic.makeRDM1(civec[i],info)
      rdm1t,rdm1a,rdm1b = fsbasic.spatialRDM(rdm1)
      print 'ieig=',i,numpy.trace(rdm1t),(numpy.trace(rdm1a),numpy.trace(rdm1b)),\
	    ' eig=',numpy.dot(fsbasic.genHVec(civec[i],info),civec[i])
   
   rdm1 = fsbasic.makeRDM1(civec[5],info)
   rdm1t,rdm1a,rdm1b = fsbasic.spatialRDM(rdm1)
   print 'rdm1t=\n',rdm1t
   print 'rdm1a=\n',rdm1a
   print 'rdm1b=\n',rdm1b
