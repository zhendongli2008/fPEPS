from fci import fsbasic
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
H1s = numpy.zeros((k0,k0))
#   1---3
#  /   /
# 0---2
hlst = [[0,1],[0,2],[1,3],[2,3]]
for i,j in hlst:
   H1s[i,j] = H1s[j,i] = t
V2s = numpy.zeros((k0,k0,k0,k0))
for i in range(k0):
   V2s[i,i,i,i] = U
H1,V2 = genSpinOrbIntegrals(H1s,V2s)

import scipy.linalg
e,v = scipy.linalg.eigh(H1)
print 'e=',e

info = [k,ndim,E0,H1,V2]

civec = numpy.load('data/vcoeff2by2.npy')
# In the bit representation, the ordering is from right to left !!!
reverse = range(k0)[-1::-1]
print 'reverse=',reverse
civec = civec.reshape([4]*k0).transpose(reverse)
civec = civec.flatten()

pp = civec.dot(civec)
php = numpy.dot(fsbasic.genHVec(civec,info),civec)
print 'ndim=',ndim
print 'pp=',pp
print 'php=',php
print 'e=',php/pp

elst = []
for i in range(k0):
   V2s = numpy.zeros((k0,k0,k0,k0))
   V2s[i,i,i,i] = U
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   info = [k,ndim,E0,H1,V2]
   eloc = numpy.dot(fsbasic.genHVec2(civec,info),civec) 
   print 'i=',i,' <Ui>=',eloc #,eloc/pp
   elst.append(eloc) 

for i,j in hlst:
   H1s = numpy.zeros((k0,k0))
   H1s[i,j] = H1s[j,i] = t
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   info = [k,ndim,E0,H1,V2]
   eloc = numpy.dot(fsbasic.genHVec1(civec,info),civec) 
   print 'i,j=',(i,j),' t<ij>=',eloc #,eloc/pp
   elst.append(eloc) 

print 'php_sum=',numpy.sum(elst)
exit()

print '\nOnly alpha-alpha (i<j) ...'
hlst = [[i,j] for i in range(k0) for j in range(k0)]
print 'hlst=',hlst
for i,j in hlst:
   # alpha-alpha
   H1s = numpy.zeros((k0,k0))
   H1s[i,j] = t
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   H1[1::2,1::2] = 0.0
   info = [k,ndim,E0,H1,V2]
   eloc = 2*numpy.dot(fsbasic.genHVec1(civec,info),civec)
   print 'i,j=',(i,j),' t<ij>[A]=',eloc,eloc/pp
   eij = eloc

   H1s = numpy.zeros((k0,k0))
   H1s[i,j] = t 
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   H1[0::2,0::2] = 0.0
   info = [k,ndim,E0,H1,V2]
   eloc = 2*numpy.dot(fsbasic.genHVec1(civec,info),civec) 
   print 'i,j=',(i,j),' t<ij>[B]=',eloc,eloc/pp
   eij += eloc
   print 'total ij=',eij

# <N>
print '\n<N>:'
H1s = numpy.zeros((k0,k0))
acc = 0.
for i in range(k0): 
   H1s[i,i] = 1.0 
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   info = [k,ndim,E0,H1,V2]
   eloc = numpy.dot(fsbasic.genHVec1(civec,info),civec)
   print 'i=',i,'<Ni>=',eloc
   H1s[i,i] = 0.0 
   acc += eloc
print 'Naveraged=',eloc,eloc/pp

# <N> from RDMs
civec = civec/numpy.linalg.norm(civec)
rdm1 = fsbasic.makeRDM1(civec,info)
rdm1t,rdm1a,rdm1b = fsbasic.spatialRDM(rdm1)
print 'rdm1t=',numpy.trace(rdm1t),'\n',rdm1t
print 'rdm1a=',numpy.trace(rdm1a),'\n',rdm1a
print 'rdm1b=',numpy.trace(rdm1b),'\n',rdm1b

# Population
print '\nPopulations'
t = civec.reshape([2]*8)
pdic = {}
for ix in numpy.ndindex(t.shape):
   n = sum(ix)
   if n not in pdic:
      pdic[n] = t[ix]**2
   else:
      pdic[n] += t[ix]**2

acc = 0.
nav = 0.
for n in range(9):
   print 'n=',n,'p=',pdic[n]
   acc += pdic[n]
   nav += n*pdic[n]
print 'ptot=',acc
assert abs(acc-1.0)<1.e-4
print 'nav =',nav
