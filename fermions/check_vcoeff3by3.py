from fci import hubbard
from fci import bits
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
k0 = 9
k = 2*k0
ndim = 2**k
E0 = 0.0
H1s = numpy.zeros((k0,k0))
#     
#     2--*---5--*---8 
#    /   |  /   |  /
#   1----*-4----*-7
#  /     |/     |/
# 0------3------6
#
hlst = [[0,1],[0,2],[3,4],[4,5],[6,7],[7,8],\
	[0,3],[3,6],[1,4],[4,7],[2,5],[5,8]]
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

civec = numpy.load('data/vcoeff3by3.npy')
# In the bit representation, the ordering is from right to left !!!
reverse = range(k0)[-1::-1]
civec = civec.reshape([4]*k0).transpose(reverse)
civec = civec.flatten()
print civec.shape

pp = civec.dot(civec)
print 'ndim=',ndim
print 'pp=',pp

# Population
print '\nPopulations'
ts = civec.reshape([2]*k)
pdic = {}
for ix in numpy.ndindex(ts.shape):
   n = sum(ix)
   if n not in pdic:
      pdic[n] = ts[ix]**2
   else:
      pdic[n] += ts[ix]**2

acc = 0.
nav = 0.
for n in range(k+1):
   print 'n=',n,'p=',pdic[n],pdic[n]/pp
   acc += pdic[n]/pp
   nav += n*pdic[n]
print 'ptot=',acc
assert abs(acc-1.0)<1.e-4
print 'nav =',nav,nav/pp

info = [k,ndim,hlst,t,U]

#==================================================================
# <Nloc>=<EiiA+EiiB>
#==================================================================
ifNloc = False
if ifNloc:
   
   def genNloc(vec,info,i):
      [k,ndim,hlst,t,U]=info
      sigma1a = numpy.zeros(ndim)
      for idx in range(ndim):
         s = 2*i
         res1 = bits.action(idx,s,0)
         if res1 == None: continue
         idx1,sgn1 = res1
         r = 2*i
         res2 = bits.action(idx1,r,1)
         if res2 == None: continue
         idx2,sgn2 = res2
         sigma1a[idx2] += sgn1*sgn2*vec[idx]
      sigma1b = numpy.zeros(ndim)
      for idx in range(ndim):
         # U*aA^+*aA*aB^+*aB = U*aA^+*aB^+*aB*aA
         s = 2*i+1
         res1 = bits.action(idx,s,0)
         if res1 == None: continue
         idx1,sgn1 = res1
         r = 2*i+1
         res2 = bits.action(idx1,r,1)
         if res2 == None: continue
         idx2,sgn2 = res2
         sigma1b[idx2] += sgn1*sgn2*vec[idx]
      sigma = sigma1a+sigma1b
      #print 'i=',i,[sigma1a.dot(vec),sigma1b.dot(vec),sigma.dot(vec)]
      return sigma.dot(vec)
   
   print '\nOccupation of spatial orbitals (fromRDM):'
   acc = 0.
   for i in range(k0):
      tmp = numpy.moveaxis(ts,[2*i,2*i+1],[0,1])
      tmp = tmp.reshape((4,-1))
      den = numpy.einsum('ik,jk->ij',tmp,tmp)
      den = numpy.diag(den)
      nloc = den.dot([0,1,1,2])
      print 'i=',i,nloc,nloc/pp
      acc += nloc
   print 'acc=',acc,acc/pp
   
   # Reversed since in the bit representation, i is counted from the last site !!!
   print '\nOccupation of spatial orbitals:'
   acc = 0.
   for i in range(k0):
      nloc = genNloc(civec,info,i)
      print 'i=',i,nloc,nloc/pp
      acc += nloc
   print 'acc=',acc,acc/pp
  
etot = 0.  
#==================================================================
# U-term
#==================================================================
ifUterm = True
if ifUterm:
   
   def genHVec2(vec,info,i):
      debug = False 
      if debug: print '[genHVec2]: 2e terms'
      [k,ndim,hlst,t,U]=info
      sigma = numpy.zeros(ndim)
      for idx in range(ndim):
         # V2: v[p,q,r,s]ap^+ aq^+ ar as 
         # U*aA^+*aA*aB^+*aB = U*aA^+*aB^+*aB*aA
         s = 2*i
         res1 = bits.action(idx,s,0)
         if res1 == None: continue
         idx1,sgn1 = res1
         r = 2*i+1
         res2 = bits.action(idx1,r,0)
         if res2 == None: continue
         idx2,sgn2 = res2
         q = 2*i+1
         res3 = bits.action(idx2,q,1)
         if res3 == None: continue
         idx3,sgn3 = res3
         p = 2*i
         res4 = bits.action(idx3,p,1)
         if res4 == None: continue
         idx4,sgn4 = res4
         sigma[idx4] += U*sgn1*sgn2*sgn3*sgn4*vec[idx]
      return sigma
   
   for i in range(k0):
      eloc = numpy.dot(genHVec2(civec,info,i),civec) 
      print 'i=',i,' <Ui>=',eloc #pp,eloc,eloc/pp
      etot += eloc

#==================================================================
# t-term
#==================================================================
ifTterm = True
if ifTterm:

   def genHVec1(vec,info,i,j):
      debug = False 
      if debug: print '[genHVec1]: 1e terms'
      [k,ndim,hlst,t,U]=info
      sigma = numpy.zeros(ndim)
      for idx in range(ndim):
          pqlst = [[2*i,2*j],[2*i+1,2*j+1],[2*j,2*i],[2*j+1,2*i+1]]
          for p,q in pqlst:
             res1 = bits.action(idx,q,0)
             if res1 == None: continue
             idx1,sgn1 = res1
             res2 = bits.action(idx1,p,1)
             if res2 == None: continue
             idx2,sgn2 = res2
             sigma[idx2] += t*sgn1*sgn2*vec[idx] 
      return sigma

   #     
   #     2--*---5--*---8 
   #    /   |  /   |  /
   #   1----*-4----*-7
   #  /     |/     |/
   # 0------3------6
   #
   print '\nVbonds'
   vbonds = [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8]]
   for i,j in vbonds:
      eloc = numpy.dot(genHVec1(civec,info,i,j),civec) 
      print 'i,j=',(i,j),' t<ij>=',eloc #pp,eloc,eloc/pp
      etot += eloc

   print '\nHbonds'
   hbonds = [[0,3],[3,6],[1,4],[4,7],[2,5],[5,8]]
   for i,j in hbonds:
      eloc = numpy.dot(genHVec1(civec,info,i,j),civec) 
      print 'i,j=',(i,j),' t<ij>=',eloc #pp,eloc,eloc/pp
      etot += eloc

print pp,etot,etot/pp
exit()

#==================================================================
# Other checks
#==================================================================
print 'php_sum=',numpy.sum(elst)

print '\nOnly alpha-alpha (i<j) ...'
for i,j in hlst:
   # alpha-alpha
   H1s = numpy.zeros((k0,k0))
   H1s[i,j] = t 
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   H1[1::2,1::2] = 0.0
   info = [k,ndim,E0,H1,V2]
   eloc = numpy.dot(fsbasic.genHVec1(civec,info),civec)
   print 'i,j=',(i,j),' t<ij>=',eloc,eloc/pp
   eij = eloc

   H1s = numpy.zeros((k0,k0))
   H1s[i,j] = t 
   H1,V2 = genSpinOrbIntegrals(H1s,V2s)
   H1[0::2,0::2] = 0.0
   info = [k,ndim,E0,H1,V2]
   eloc = numpy.dot(fsbasic.genHVec1(civec,info),civec) 
   print 'i,j=',(i,j),' t<ij>=',eloc,eloc/pp
   eij += eloc
   print 'total ij=',eij*2

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
