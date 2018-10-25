#
# Simplest FCI
#
import numpy
import bits
import dvdson

# Given Slater det |orblst>, calculate energy of it E[|orblst>]
# Hlocal = hii ai^+ ai + viiii ai^+ ai^+ ai ai
def elocal(orblst,E0,H1,V2):
   e1e=numpy.sum([H1[i,i] for i in orblst])
   e2e=-numpy.sum([[V2[i,j,i,j] for i in orblst] for j in orblst])
   etot=e1e+e2e+E0
   return etot

def genDiag(info):
   debug = False	
   [k,ndim,E0,H1,V2]=info
   if debug: print '\n[genDiag] norb/ndim=',k,2**k
   diag = numpy.zeros(ndim)
   for idx in range(ndim):
      orblst = bits.toOrblst(k,idx)
      diag[idx] = elocal(orblst,E0,H1,V2)
      if debug: print ' idx=',idx,bin(idx),bits.bitCount(idx),orblst,diag[idx]
   return diag

#
# H*Vec
#
def genHVec(vec,info):
   return genHVec1(vec,info)\
 	 +genHVec2(vec,info)

def genHVec1(vec,info):
   debug = False 
   if debug: print '[genHVec1]: 1e terms'
   [k,ndim,E0,H1,V2]=info
   sigma = numpy.zeros(ndim)
   for idx in range(ndim):
      # H1: h[p,q]ap^+aq
      for q in range(k):
         res1 = bits.action(idx,q,0)
  	 if res1 == None: continue
	 idx1,sgn1 = res1
         for p in range(k):
   	    res2 = bits.action(idx1,p,1)
  	    if res2 == None: continue
	    idx2,sgn2 = res2
	    sigma[idx2] += H1[p,q]*sgn1*sgn2*vec[idx] 
   return sigma

def genHVec2(vec,info):
   debug = False 
   if debug: print '[genHVec2]: 2e terms'
   [k,ndim,E0,H1,V2]=info
   sigma = numpy.zeros(ndim)
   for idx in range(ndim):
      # V2: v[p,q,r,s]ap^+ aq^+ ar as
      for s in range(k):
         res1 = bits.action(idx,s,0)
	 if res1 == None: continue
	 idx1,sgn1 = res1
         for r in range(k):
   	    res2 = bits.action(idx1,r,0)
  	    if res2 == None: continue
	    idx2,sgn2 = res2
            for q in range(k):
              res3 = bits.action(idx2,q,1)
     	      if res3 == None: continue
     	      idx3,sgn3 = res3
              for p in range(k):
        	 res4 = bits.action(idx3,p,1)
       	         if res4 == None: continue
     	         idx4,sgn4 = res4
	         sigma[idx4] += V2[p,q,r,s]*sgn1*sgn2*sgn3*sgn4*vec[idx]
   return sigma

#
# Generate RDMs
#
def makeRDM1(civec,info):
   debug = False
   [k,ndim,E0,H1,V2]=info
   if debug: print '\n[makeRDM1] norb/ndim=',k,2**k
   rdm1 = numpy.zeros((k,k))
   for idx in range(ndim):
      # ap^+aq
      for q in range(k):
         res1 = bits.action(idx,q,0)
  	 if res1 == None: continue
	 idx1,sgn1 = res1
         for p in range(k):
   	    res2 = bits.action(idx1,p,1)
  	    if res2 == None: continue
	    idx2,sgn2 = res2
	    # g[p,q] += C^T[idx2]*<idx2|ap^+aq|idx>*C[idx]
	    rdm1[p,q] += sgn1*sgn2*civec[idx2]*civec[idx] 
   return rdm1

# <i1^+i2^+j2j1>
def makeRDM2(civec,info):
   debug = False
   [k,ndim,E0,H1,V2]=info
   if debug: print '\n[makeRDM2] norb/ndim=',k,2**k
   rdm2 = numpy.zeros((k,k,k,k))
   for idx in range(ndim):
      # ap^+aq^+aras
      for s in range(k):
         res1 = bits.action(idx,s,0)
	 if res1 == None: continue
	 idx1,sgn1 = res1
         for r in range(k):
   	    res2 = bits.action(idx1,r,0)
  	    if res2 == None: continue
	    idx2,sgn2 = res2
            for q in range(k):
              res3 = bits.action(idx2,q,1)
     	      if res3 == None: continue
     	      idx3,sgn3 = res3
              for p in range(k):
        	 res4 = bits.action(idx3,p,1)
       	         if res4 == None: continue
     	         idx4,sgn4 = res4
	         rdm2[p,q,r,s] += sgn1*sgn2*sgn3*sgn4*civec[idx]*civec[idx4]
   # G[i1,i2;j1,j2] = <i1^+ i2^+ j2 j1>
   rdm2 = -rdm2
   return rdm2

#
# Generate p-RDMs
#
def makePRDM1(civec,info):
   debug = False
   [k,ndim,E0,H1,V2,S1,R3,P2]=info
   if debug: print '\n[makePRDM1] norb/ndim=',k,2**k
   rdm1 = numpy.zeros(k)
   for idx in range(ndim):
      # ap^+
      for p in range(k):
         res1 = bits.action(idx,p,1)
  	 if res1 == None: continue
	 idx1,sgn1 = res1
	 rdm1[p] += sgn1*civec[idx1]*civec[idx] 
   return rdm1

def spatialRDM(rdm1):
   rdm1a = rdm1[::2,::2]
   rdm1b = rdm1[1::2,1::2]
   rdm1t = rdm1a+rdm1b
   return rdm1t,rdm1a,rdm1b

#
# FS-FCI solver
#
def fsFCI(info,nroot=1,crit_e=1.e-10,crit_vec=1.e-6):
   # Davidson Object 
   Diag = genDiag(info)
   masker = dvdson.mask(info,genHVec)
   # Solve      
   ndim = 2**info[0]
   solver = dvdson.eigenSolver()
   solver.iprt = 0
   solver.crit_e   = crit_e
   solver.crit_vec = crit_vec
   solver.ndim = ndim 
   solver.diag = Diag
   solver.neig = min(ndim,nroot)
   solver.matvec = masker.matvec
   solver.noise = True
   eigs,civec,nmvp = solver.solve_iter()
   return eigs,civec
